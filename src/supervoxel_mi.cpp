/**
 * @file supervoxel_mi.cpp
 *
 *
 * @author Nikhil Mehta
 * @date 2016-11-21
 */

#include <string>
#include <boost/thread/thread.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/octree/octree_pointcloud_adjacency.h>

#include "supervoxel_cluster_search.h"
#include "supervoxel_mapping.hpp"

using namespace pcl;
using namespace std;

typedef PointXYZRGBA PointT;
typedef PointCloud<PointT> PointCloudT;
typedef SupervoxelClustering<PointT> SupervoxelClusteringT;
typedef PointXYZL PointLT;
typedef PointCloud<PointLT> PointLCloudT;

struct options {

	float vr;
	float sr;
	float colorWeight;
	float spatialWeight;
	float normalWeight;
	int test;

}programOptions;

void
showPointCloud(typename PointCloudT::Ptr);

void
showPointClouds(PointCloudT::Ptr, PointCloudT::Ptr);

int initOptions(int argc, char* argv[]) {

	namespace po = boost::program_options;

	programOptions.colorWeight = 0.0f;
	programOptions.spatialWeight = 0.4f;
	programOptions.normalWeight = 1.0f;
	programOptions.vr = 1.0f;
	programOptions.sr = 5.0f;
	programOptions.test = 324;

	po::options_description desc ("Allowed Options");
	desc.add_options()
													("help,h", "Usage <Scan 1 Path> <Scan 2 Path> <Transform File>")
													("voxel_res,v", po::value<float>(&programOptions.vr), "voxel resolution")
													("seed_res,s", po::value<float>(&programOptions.sr), "seed resolution")
													("color_weight,c", po::value<float>(&programOptions.colorWeight), "color weight")
													("spatial_weight,z", po::value<float>(&programOptions.spatialWeight), "spatial weight")
													("normal_weight,n", po::value<float>(&programOptions.normalWeight), "normal weight")
													("test,t", po::value<int>(&programOptions.test), "test");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		cout << "Supervoxel MI based SCan Alignment" << endl << desc << endl;
		return 1;
	} else {

		cout << "vr: " << programOptions.vr << endl;
		cout << "sr: " << programOptions.sr << endl;
		cout << "colorWeight: " << programOptions.colorWeight << endl;
		cout << "spatialWeight: " << programOptions.spatialWeight << endl;
		cout << "normalWeight: " << programOptions.normalWeight << endl;

		return 0;
	}

}


int
main (int argc, char *argv[]) {

	if (initOptions(argc, argv))
		return 1;

	if (argc < 3) {
		cerr << "One or more scan files/transform missing";
		return 1;
	}

	int s1, s2;
	string transformFile;

	s1 = atoi(argv[1]);
	s2 = atoi(argv[2]);
	transformFile = argv[3];

	const string dataDir = "../../Data/scans_pcd/scan_";

	PointCloudT::Ptr scan1 = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudT::Ptr scan2 = boost::shared_ptr <PointCloudT> (new PointCloudT ());
	PointCloudT::Ptr temp = boost::shared_ptr <PointCloudT> (new PointCloudT ());

	cout<<"Loading PointClouds..."<<endl;

	stringstream ss;

	ss << dataDir << boost::format("%04d.pcd")%s1;

	if (io::loadPCDFile<PointT> (ss.str(), *scan1)) {
		cout << "Error loading cloud file: " << ss.str() << endl;
		return (1);
	}

	ss.str(string());
	ss << dataDir << boost::format("%04d.pcd")%s2;

	if (io::loadPCDFile<PointT> (ss.str(), *temp)) {
		cout << "Error loading cloud file: " << ss.str() << endl;
		return (1);
	} else {

		ifstream in(transformFile.c_str());
		if (!in) {
			stringstream err;
			err << "Error loading transformation " << transformFile.c_str() << endl;
			cerr<<err.str();
			return 1;
		}

		Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
		string line;

		for (int i = 0; i < 4; i++) {
			std::getline(in,line);

			std::istringstream sin(line);
			for (int j = 0; j < 4; j++) {
				sin >> transform (i,j);
			}
		}

		transformPointCloud (*temp, *scan2, (Eigen::Matrix4f) transform.inverse());
		temp->clear();
	}

	// scan1 wrt
	//showPointClouds(scan1, scan2);

	SupervoxelClusteringT super (programOptions.vr, programOptions.sr);

	super.setInputCloud(scan1);
	super.setColorImportance(programOptions.colorWeight);
	super.setSpatialImportance(programOptions.spatialWeight);
	super.setNormalImportance(programOptions.normalWeight);

	map <uint32_t, Supervoxel<PointT>::Ptr> supervoxelClusters;
	cout << "Extracting Supervoxels" << endl;

	// Time
	clock_t start = clock();

	super.extract(supervoxelClusters);

	clock_t end = clock();
	double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

	PointCloud<PointXYZL>::Ptr temoCloud = super.getLabeledCloud();
	PointCloud<PointXYZL>::iterator itr = temoCloud->begin();

	map<uint, typename SuperVoxelMapping::Ptr> SVMapping;
	unsigned int cloudCtr = 0;
	for (; itr!=temoCloud->end(); ++itr, ++cloudCtr) {

		PointXYZL p = (*itr);

		if (p.label == 0)
			continue;

		if (SVMapping.find(p.label) != SVMapping.end()) {
			SVMapping[p.label]->getScanAIndices()->push_back(cloudCtr);
		} else {
			typename SuperVoxelMapping::Ptr newPtr (new SuperVoxelMapping(p.label));
			SVMapping.insert(pair<uint, typename SuperVoxelMapping::Ptr>(p.label, newPtr));
			SVMapping[p.label]->getScanAIndices()->push_back(cloudCtr);
		}

	}

	temoCloud->clear();

	map<typename SupervoxelClustering<PointT>::LeafContainerT*, uint32_t> labeledLeafMap;
	super.getLabeledLeafContainerMap(labeledLeafMap);

	pcl::SupervoxelClustering<PointT>::OctreeAdjacencyT::Ptr adjTree = super.getOctreeeAdjacency();

	PointCloud<PointT>::iterator scanItr = scan2->begin();
	int scan2Counter = 0;

	int totalPointInScan1Voxels = 0;

	for (;scanItr != scan2->end(); ++scanItr, ++scan2Counter) {

		PointT a = (*scanItr);


		bool presentInVoxel = adjTree -> isVoxelOccupiedAtPoint(a);

		if (presentInVoxel) {

			totalPointInScan1Voxels++;

			typename SupervoxelClustering<PointT>::LeafContainerT* leaf = adjTree -> getLeafContainerAtPoint(a);

			// label of supervoxel
			if (labeledLeafMap.find(leaf) != labeledLeafMap.end()) {

				unsigned int label = labeledLeafMap[leaf];

				if (SVMapping.find(label) != SVMapping.end()) {
					SVMapping[label]->getScanBIndices()->push_back(scan2Counter);
				}
			}
		}
	}

	// Display 324

	int SV = programOptions.test;

	typename PointCloudT::Ptr newCloud (new PointCloudT);

	vector<unsigned int>::iterator vi = SVMapping[SV] -> getScanAIndices() -> begin();

	for (; vi != SVMapping[SV] -> getScanAIndices() -> end(); ++vi) {

		PointT p = scan1->at(*vi);

		p.r = 255;
		p.g = 0;
		p.b = 0;

		newCloud->push_back(p);
	}

	vi = SVMapping[SV] -> getScanBIndices() -> begin();

	for (; vi != SVMapping[SV] -> getScanBIndices() -> end(); ++vi) {

		PointT p = scan2->at(*vi);

		p.r = 0;
		p.g = 255;
		p.b = 0;

		newCloud->push_back(p);
	}

	cout << boost::format("%d points out of %d of scan2 are present in %d voxels of scan1")%totalPointInScan1Voxels%scan2->size()%adjTree->getLeafCount() << endl;
	cout << boost::format("Found %d supervoxels in %f ")%supervoxelClusters.size()%time_spent << endl;

	showPointCloud(newCloud);

//	map<uint, typename SuperVoxelMapping::Ptr>::iterator svItr = SVMapping.begin();
//	for (; svItr!=SVMapping.end(); ++svItr) {
//
//		// Write MI Code
//
//		typename SuperVoxelMapping::Ptr svm = svItr->second;
//
//		boost::shared_ptr<vector<unsigned int> > scanAIndices = svm -> getScanAIndices();
//		boost::shared_ptr<vector<unsigned int> > scanBIndices = svm -> getScanBIndices();
//
//
//
//		cout << boost::format("Label: %d -	%d	%d")%svItr->first%(svItr->second)->getScanAIndices()->size()%(svItr->second)->getScanBIndices()->size() << endl;
//	}

}

void
showPointCloud(PointCloudT::Ptr scan1, PointCloudT::Ptr scan2) {

	boost::shared_ptr<visualization::PCLVisualizer> viewer (new visualization::PCLVisualizer ("Supervoxel Based MI Viewer"));
	viewer->setBackgroundColor (0,0,0);

	string id1("scan1"), id2("scan2");

	visualization::PointCloudColorHandlerRGBField<PointT> rgb1(scan1);
	viewer->addPointCloud<PointT> (scan1, rgb1, id1);

	visualization::PointCloudColorHandlerRGBField<PointT> rgb2(scan2);
	viewer->addPointCloud<PointT> (scan2, rgb2, id2);

	viewer->setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE, 3, id1);
	viewer->setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE, 3, id2);
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters();

	while(!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep (boost::posix_time::microseconds (1e5));
	}

}

void
showPointCloud(typename PointCloudT::Ptr scan) {

	boost::shared_ptr<visualization::PCLVisualizer> viewer (new visualization::PCLVisualizer ("Supervoxel Based MI Viewer"));
	viewer->setBackgroundColor (0,0,0);

	string id1("scan");

	visualization::PointCloudColorHandlerRGBField<PointT> rgb1(scan);
	viewer->addPointCloud<PointT> (scan, rgb1, id1);

	viewer->setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE, 3, id1);
	viewer->addCoordinateSystem (1.0);
	viewer->initCameraParameters();

	while(!viewer->wasStopped()) {
		viewer->spinOnce(100);
		boost::this_thread::sleep (boost::posix_time::microseconds (1e5));
	}

}



















