#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <std_srvs/Empty.h>


#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
using namespace cv;
using namespace std;
using namespace sensor_msgs;
using namespace message_filters;


float fx = 1310.0;

float fy = 1310.0;

float cx = 612.0;

float cy = 512.0;

float bf = 235.745;

image_transport::Publisher depth_img_pub;

image_transport::Publisher depth_conf_img_pub;

double LastTimeStamp = 0.0;

double TimeStep = 1.0;

map<double, Eigen::Isometry3d> Stamped_Poses;

Eigen::Isometry3d T_b_c = Eigen::Isometry3d::Identity();

pcl::PointCloud<pcl::PointXYZRGB> global_map;

struct DepthEstParamters
{
	float _P1;
	float _P2;
	int _correlation_window_size;
	int _disp12MaxDiff;
	int _disparity_range;
	int _min_disparity;
	int _prefilter_cap;
	int _prefilter_size;
	int _speckle_range;
	int _speckle_size;
	int _texture_threshold;
	float _uniqueness_ratio;
	DepthEstParamters()
	{};
	DepthEstParamters(float P1,
					  float P2,
					  int correlation_window_size,
					  int disp12MaxDiff,
					  int disparity_range,
					  int min_disparity,
					  int prefilter_cap,
					  int prefilter_size,
					  int speckle_range,
					  int speckle_size,
					  int texture_threshold,
					  float uniqueness_ratio)
		: _P1(P1), _P2(P2), _correlation_window_size(correlation_window_size),
		  _disp12MaxDiff(disp12MaxDiff),
		  _disparity_range(disparity_range), _min_disparity(min_disparity),
		  _prefilter_cap(prefilter_cap), _prefilter_size(prefilter_size), _speckle_range(speckle_range),
		  _speckle_size(speckle_size), _texture_threshold(texture_threshold), _uniqueness_ratio(uniqueness_ratio)
	{};

};

DepthEstParamters param;

std::string getImageType(int number)
{
	// find type
	int imgTypeInt = number % 8;
	std::string imgTypeString;

	switch (imgTypeInt) {
	case 0: imgTypeString = "8U";
		break;
	case 1: imgTypeString = "8S";
		break;
	case 2: imgTypeString = "16U";
		break;
	case 3: imgTypeString = "16S";
		break;
	case 4: imgTypeString = "32S";
		break;
	case 5: imgTypeString = "32F";
		break;
	case 6: imgTypeString = "64F";
		break;
	default: break;
	}

	// find channel
	int channel = (number / 8) + 1;

	std::stringstream type;
	type << "CV_" << imgTypeString << "C" << channel;

	return type.str();
}

inline Eigen::Vector3d project2Dto3D(int x, int y, float d, float fx, float fy, float cx, float cy, float fb)
{
	// float zz = float(d) / scale;
	float zz = fb / d;
	float xx = zz * (x - cx) / fx;
	float yy = zz * (y - cy) / fy;
	return Eigen::Vector3d(xx, yy, zz);
}

void computeDisp(Mat left, Mat right, Mat &out, Mat &out_conf, const DepthEstParamters &param, bool visualize)
{
	using namespace ximgproc;
	String dst_path = "/home/da/project/ros/catkin_ws/src/sparse_direct_vo/data/disp.png";
	String dst_raw_path = "None";
	String dst_conf_path = "None";
	String algo = "bm";
	String filter = "wls_conf";
	bool no_display = !visualize;
	bool no_downscale = true;
	int max_disp = 128;
	double lambda = 8000;
	double sigma = 1.5;
	double prefilter_cap = 31;
	double prefilter_size = 255;
	// speckle_range: 31
	// speckle_size: 300
	double speckle_range = 31;
	double speckle_size = 300;
	double uniqueness_ratio = 10.0;

//	int wsize = 13;

	Mat left_for_matcher, right_for_matcher;
	Mat left_disp, right_disp;
	Mat filtered_disp, solved_disp, solved_filtered_disp;
	Mat conf_map = Mat(left.rows, left.cols, CV_8U);
	conf_map = Scalar(255);
	Rect ROI;
	Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
	double matching_time, filtering_time;
	double solving_time = 0;
	if (max_disp <= 0 || max_disp % 16 != 0) {
		cout << "Incorrect max_disparity value: it should be positive and divisible by 16";
		return;
	}
	if (param._correlation_window_size <= 0 || param._correlation_window_size % 2 != 1) {
		cout << "Incorrect window_size value: it should be positive and odd";
		return;
	}

	if (filter == "wls_conf") // filtering with confidence (significantly better quality than wls_no_conf)
	{
		if (!no_downscale) {
			// downscale the views to speed-up the matching stage, as we will need to compute both left
			// and right disparity maps for confidence map computation
			//! [downscale]
			max_disp /= 2;
			if (max_disp % 16 != 0) {
				max_disp += 16 - (max_disp % 16);
			}
			// resize(left ,left_for_matcher ,Size(),0.5,0.5, INTER_LINEAR_EXACT);
			// resize(right,right_for_matcher,Size(),0.5,0.5, INTER_LINEAR_EXACT);
			resize(left, left_for_matcher, Size(), 0.5, 0.5, INTER_LINEAR);
			resize(right, right_for_matcher, Size(), 0.5, 0.5, INTER_LINEAR);
			//! [downscale]
		}
		else {
			left_for_matcher = left.clone();
			right_for_matcher = right.clone();
		}

		if (algo == "bm") {
			//! [matching]
			Ptr<StereoBM> left_matcher = StereoBM::create(param._disparity_range, param._correlation_window_size);
			left_matcher->setDisp12MaxDiff(param._disp12MaxDiff);
			left_matcher->setMinDisparity(param._min_disparity);
			left_matcher->setPreFilterCap(param._prefilter_cap);
			left_matcher->setPreFilterSize(param._prefilter_size);
			left_matcher->setSpeckleRange(param._speckle_range);
			left_matcher->setSpeckleWindowSize(param._speckle_size);
			left_matcher->setTextureThreshold(param._texture_threshold);
			left_matcher->setUniquenessRatio(param._uniqueness_ratio);
			wls_filter = createDisparityWLSFilter(left_matcher);
			Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

			if (left_for_matcher.channels() == 3 || left_for_matcher.channels() == 4) {
				cvtColor(left_for_matcher, left_for_matcher, COLOR_BGR2GRAY);
				cvtColor(right_for_matcher, right_for_matcher, COLOR_BGR2GRAY);
			}


			matching_time = (double)getTickCount();
			left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
			right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
			matching_time = ((double)getTickCount() - matching_time) / getTickFrequency();
			//! [matching]
		}
		else if (algo == "sgbm") {
			Ptr<StereoSGBM> left_matcher =
				StereoSGBM::create(param._min_disparity, param._disparity_range, param._correlation_window_size);
			left_matcher->setP1(param._P1);
			left_matcher->setP2(param._P2);
			left_matcher->setPreFilterCap(param._prefilter_cap);
			left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
			wls_filter = createDisparityWLSFilter(left_matcher);
			Ptr<StereoMatcher> right_matcher = createRightMatcher(left_matcher);

			matching_time = (double)getTickCount();
			left_matcher->compute(left_for_matcher, right_for_matcher, left_disp);
			right_matcher->compute(right_for_matcher, left_for_matcher, right_disp);
			matching_time = ((double)getTickCount() - matching_time) / getTickFrequency();
		}
		else {
			cout << "Unsupported algorithm";
			return;
		}

		//! [filtering]
		wls_filter->setLambda(lambda);
		wls_filter->setSigmaColor(sigma);
		filtering_time = (double)getTickCount();
		wls_filter->filter(left_disp, left, filtered_disp, right_disp);
		filtering_time = ((double)getTickCount() - filtering_time) / getTickFrequency();
		//! [filtering]
		conf_map = wls_filter->getConfidenceMap();

		// Get the ROI that was used in the last filter call:
		ROI = wls_filter->getROI();
		if (!no_downscale) {
			// upscale raw disparity and ROI back for a proper comparison:
			resize(left_disp, left_disp, Size(), 2.0, 2.0, INTER_LINEAR);
			left_disp = left_disp * 2.0;
			ROI = Rect(ROI.x * 2, ROI.y * 2, ROI.width * 2, ROI.height * 2);
		}
	}
	Mat filtered_disp_vis;
	getDisparityVis(filtered_disp, filtered_disp_vis, 1.0);
//	out = filtered_disp;
	out = filtered_disp_vis.clone();
	out_conf = conf_map.clone();


	if (!no_display) {
		namedWindow("left", WINDOW_AUTOSIZE);
		imshow("left", left);
		namedWindow("right", WINDOW_AUTOSIZE);
		imshow("right", right);
		namedWindow("confidence_map", WINDOW_AUTOSIZE);
		imshow("confidence_map", conf_map);

		//! [visualization]
		Mat raw_disp_vis;
		getDisparityVis(left_disp, raw_disp_vis, 1.0);
		namedWindow("raw disparity", WINDOW_AUTOSIZE);
		imshow("raw disparity", raw_disp_vis);
		Mat filtered_disp_vis;
		getDisparityVis(filtered_disp, filtered_disp_vis, 1.0);
		namedWindow("filtered disparity", WINDOW_AUTOSIZE);
		imshow("filtered disparity", filtered_disp_vis);

		if (!solved_disp.empty()) {
			Mat solved_disp_vis;
			getDisparityVis(solved_disp, solved_disp_vis, 1.0);
			namedWindow("solved disparity", WINDOW_AUTOSIZE);
			imshow("solved disparity", solved_disp_vis);

			Mat solved_filtered_disp_vis;
			getDisparityVis(solved_filtered_disp, solved_filtered_disp_vis, 1.0);
			namedWindow("solved wls disparity", WINDOW_AUTOSIZE);
			imshow("solved wls disparity", solved_filtered_disp_vis);
		}

		// while (1)
		// {
		//     char key = (char)waitKey();
		//     if (key == 27 || key == 'q' || key == 'Q') // 'ESC'
		//         break;
		// }
		waitKey(10);
		//! [visualization]
	}
}

void img_callback(const sensor_msgs::ImageConstPtr &left, const sensor_msgs::ImageConstPtr &right)
{
	if ((left->header.stamp.toSec() - LastTimeStamp) < TimeStep) {
//		ROS_INFO_STREAM("skip");
		return;
	}

	LastTimeStamp = left->header.stamp.toSec();
	cv_bridge::CvImagePtr cv_ptr_l;
	cv_bridge::CvImagePtr cv_ptr_r;
	try {
		cv_ptr_l = cv_bridge::toCvCopy(left, "bgr8");
		cv_ptr_r = cv_bridge::toCvCopy(right, "bgr8");
	}
	catch (cv_bridge::Exception &e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}

	cv::Mat img_l = cv_ptr_l->image.clone();
	cv::Mat img_r = cv_ptr_r->image.clone();
	double time = left->header.stamp.toSec();
	if (Stamped_Poses.find(time) == Stamped_Poses.end()) {
		ROS_WARN_STREAM("Didn't found timestamp in pose file!" << "timestamp: " << fixed << setprecision(12) << time);
		return;
	}


	cv::Mat disp, disp_conf;

	// the pose save in file is Twc(T_c0_cj)
	Eigen::Isometry3d T_b0_bj = Stamped_Poses[time];
	Eigen::Isometry3d T_b0_cj = T_b0_bj * T_b_c;
//		Eigen::Isometry3d T_c0_cmj = T_c0_cj * T_c_orb;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sub_map(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sub_map_transformed(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sub_map_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

	cv::Mat img_l_rgb, img_r_rgb;
	if (img_l.empty() || img_r.empty()) {
		ROS_WARN_STREAM("found empty images!" << "timestamp: " << fixed << setprecision(12) << time);
		return;
	}
	img_l_rgb = img_l.clone();
	img_r_rgb = img_r.clone();
	computeDisp(img_l, img_r, disp, disp_conf, param, false);
	disp_conf.convertTo(disp_conf, CV_8UC1);

//	string mat_type = getImageType(disp_conf.type());
//	cout << mat_type << endl;
	cv_bridge::CvImage img_bridge = cv_bridge::CvImage(left->header, sensor_msgs::image_encodings::MONO8, disp);
	depth_img_pub.publish(img_bridge.toImageMsg());
	img_bridge = cv_bridge::CvImage(left->header, sensor_msgs::image_encodings::MONO8, disp_conf);
	depth_conf_img_pub.publish(img_bridge.toImageMsg());

//		cv::Mat disp_f;
//		disp.convertTo(disp_f,CV_32F);

	for (int x = 0; x < img_l_rgb.cols; x++) {
		for (int y = 0; y < img_l_rgb.rows; y++) {
			float d = static_cast<float >(disp.at<unsigned char>(y, x));
			float d_f = disp_conf.at<float>(y, x);
			unsigned char b = static_cast<int>(img_l_rgb.at<Vec3b>(y, x)[0]);
			unsigned char g = static_cast<int>(img_l_rgb.at<Vec3b>(y, x)[1]);
			unsigned char r = static_cast<int>(img_l_rgb.at<Vec3b>(y, x)[2]);
//				cout << "disp confidence: " << d_f << endl;

			if (d_f <= 160 || (bf / d) > 10) {
				continue;
			}
			Eigen::Vector3d p_3d = project2Dto3D(x, y, d, fx, fy, cx, cy, bf);
//				p_3d = T_c0_cj * p_3d;
//				p_3d = T_c0_cj * p_3d;
//				cout << "depth: " << (bf / d) << endl;
			pcl::PointXYZRGB p(r, g, b);
			p.x = p_3d.x();
			p.y = p_3d.y();
			p.z = p_3d.z();

			sub_map->push_back(p);
		}
	}
	if(sub_map->points.empty())
		return;
	pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
	sor.setInputCloud(sub_map);
	sor.setMeanK(50);
	sor.setStddevMulThresh(0.1);
	sor.filter(*sub_map_filtered);
	Eigen::Vector3d t = T_b0_cj.translation();
	Eigen::Quaterniond q(T_b0_cj.rotation());
//	cout << "transformation : \nt: " << t.transpose() << "\n "
//		 << "q: " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " " << endl;
	pcl::transformPointCloud(*sub_map_filtered, *sub_map_transformed, T_b0_cj.matrix());

	global_map += *sub_map_transformed;

}

bool save(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
	std::string out_path;
//	ros::param::param<std::string>("out_path", out_path, "/home/da/project/ros/catkin_ws/src/stereo_dense_mapper/data");
	ros::param::get("/stereo_mapper_node/out_path",out_path);
	std::string out_file = out_path + "/map.pcd";
	ROS_INFO_STREAM("map saved to " + out_file);
	pcl::io::savePCDFileBinary(out_file, global_map);
	return true;
}

int main(int argc, char **argv)
{
	string img_l_topic, img_r_topic, config_file, pose_file;
	ros::init(argc, argv, "reconstruction");
	ros::NodeHandle n;


	ros::param::get("/stereo_mapper_node/config_file",config_file);
	ros::param::get("/stereo_mapper_node/pose_file",pose_file);
	ros::param::get("/stereo_mapper_node/img_l_topic",img_l_topic);
	ros::param::get("/stereo_mapper_node/img_r_topic",img_r_topic);



	image_transport::ImageTransport it(n);
	depth_img_pub = it.advertise("/reconstructor/depth", 10);
	depth_conf_img_pub = it.advertise("/reconstructor/depth_confidence", 10);
	image_transport::TransportHints hints("compressed");
	image_transport::SubscriberFilter img_l_sub(it, img_l_topic, 500, hints);
	image_transport::SubscriberFilter img_r_sub(it, img_r_topic, 500, hints);
	typedef sync_policies::ApproximateTime<Image, Image> Img_sync;
	Synchronizer<Img_sync> img_sync(Img_sync(500), img_l_sub, img_r_sub);
	img_sync.registerCallback(boost::bind(&img_callback, _1, _2));
	ros::ServiceServer service = n.advertiseService("/recontructor/save_map", save);

	fstream file;
	vector<double> vTimeStamp;
	vector<Eigen::Vector3d> vt;
	vector<Eigen::Quaterniond> vq;
	vector<Eigen::Isometry3d> vT;

	FileStorage fs(config_file, FileStorage::READ);
	FileNode node = fs["dictitems"];
	float P1 = (float)node["P1"];
	float P2 = (float)node["P2"];
	int correlation_window_size = (int)node["correlation_window_size"];
	int disp12MaxDiff = (int)node["disp12MaxDiff"];
	int disparity_range = (int)node["disparity_range"];
	int min_disparity = (int)node["min_disparity"];
	int prefilter_cap = (int)node["prefilter_cap"];
	int prefilter_size = (int)node["prefilter_size"];
	int speckle_range = (int)node["speckle_range"];
	int speckle_size = (int)node["speckle_size"];
	int texture_threshold = (int)node["texture_threshold"];
	float uniqueness_ratio = (float)node["uniqueness_ratio"];

	Mat Tbc;
	node = fs["Tbc"];
	if (!node.empty()) {
		Tbc = node.mat();
		cv::cv2eigen(Tbc, T_b_c.matrix());
	}
	param = DepthEstParamters(P1,
							  P2,
							  correlation_window_size,
							  disp12MaxDiff,
							  disparity_range,
							  min_disparity,
							  prefilter_cap,
							  prefilter_size,
							  speckle_range,
							  speckle_size,
							  texture_threshold,
							  uniqueness_ratio);

//	read Transformation of camera
	ROS_INFO_STREAM("open pose file: " + pose_file);
	file.open(pose_file, ios::in);
	if (!file) {
		cout << "fail to open file" << endl;
		return -1;
	}
	string line;
	while (getline(file, line)) {
		if (line.find('#') != -1) {
			continue;
		}
		istringstream iss(line);
		double x, y, z, w, time;
		if (!(iss >> time >> x >> y >> z)) {
			cout << "cannot read position for est" << endl;
			break;
		}
		vTimeStamp.push_back(time);
		Eigen::Vector3d t(x, y, z);
//		vt.push_back(t);
		if (!(iss >> x >> y >> z >> w)) {
			cout << "cannot read rotation for est" << endl;
			break;
		}
		Eigen::Quaterniond q(w, x, y, z);
//		vq.push_back(q);

		Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
		T.rotate(q);
		T.pretranslate(t);
		vT.push_back(T);

		Stamped_Poses.insert({time, T});
	}
	file.close();

	ros::spin();


	return 0;
}

