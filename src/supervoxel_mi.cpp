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
#include <cmath>
#include <iomanip>

using namespace pcl;
using namespace std;

typedef PointXYZRGBA PointT;
typedef PointCloud<PointT> PointCloudT;
typedef SupervoxelClustering<PointT> SupervoxelClusteringT;
typedef PointXYZL PointLT;
typedef PointCloud<PointLT> PointLCloudT;
//typedef typename std::vector<typename SuperVoxelMapping::Ptr> SVMappingVector;
typedef std::map<uint, typename SuperVoxelMappingHelper::Ptr> SVMap;
typedef std::map<typename SupervoxelClusteringT::LeafContainerT*, uint32_t> LabeledLeafMapT;
typedef typename SupervoxelClusteringT::OctreeAdjacencyT::Ptr AdjacencyOctreeT;

struct options {

	float vr;
	float sr;
	float colorWeight;
	float spatialWeight;
	float normalWeight;
	int test;

}programOptions;

void genOctreeKeyforPoint(const typename SupervoxelClusteringT::OctreeAdjacencyT::Ptr adjTree, const PointT& point_arg, SuperVoxelMappingHelper::OctreeKeyT & key_arg) {
	// calculate integer key for point coordinates

	double min_x, min_y, min_z;
	double max_x, max_y, max_z;

	adjTree -> getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
	double resolution = adjTree->getResolution();

	key_arg.x = static_cast<unsigned int> ((point_arg.x - min_x) / resolution);
	key_arg.y = static_cast<unsigned int> ((point_arg.y - min_y) / resolution);
	key_arg.z = static_cast<unsigned int> ((point_arg.z - min_z) / resolution);
}

void
showPointCloud(typename PointCloudT::Ptr);

void
showPointClouds(PointCloudT::Ptr, PointCloudT::Ptr);

void
showTestSuperVoxel(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2);

void
computeSupervoxelScan2Data(SVMap& SVMapping, PointCloudT::Ptr scan2);

void
computeSupervoxelScan1Data(SVMap& SVMapping, PointCloudT::Ptr scan1);


void
calculateMutualInformation(SVMap& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2);

void
createSuperVoxelMappingForScan1 (SVMap& SVMapping, const typename PointCloudT::Ptr scan, LabeledLeafMapT& labeledLeafMapping, const AdjacencyOctreeT& adjTree);

void
createSuperVoxelMappingForScan2 (SVMap& SVMapping, const typename PointCloudT::Ptr scan, LabeledLeafMapT& labeledLeafMapping, const AdjacencyOctreeT& adjTree);


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

	// Not being used for now
	map <uint32_t, Supervoxel<PointT>::Ptr> supervoxelClusters;
	cout << "Extracting Supervoxels" << endl;

	// Time
	clock_t start = clock();

	super.extract(supervoxelClusters);

	clock_t end = clock();
	double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

	// Original Point Cloud with sv labels
	//	PointCloud<PointXYZL>::Ptr temoCloud = super.getLabeledCloud();
	//	PointCloud<PointXYZL>::iterator itr = temoCloud->begin();

	LabeledLeafMapT labeledLeafMap;
	super.getLabeledLeafContainerMap(labeledLeafMap);

	pcl::SupervoxelClustering<PointT>::OctreeAdjacencyT::Ptr adjTree = super.getOctreeeAdjacency();

	cout << "LeafCount " << adjTree->getLeafCount() << ' ' << labeledLeafMap.size() << endl;

	SVMap SVMapping;
	createSuperVoxelMappingForScan1(SVMapping,scan1, labeledLeafMap, adjTree);
	createSuperVoxelMappingForScan2(SVMapping,scan2, labeledLeafMap, adjTree);

	//	cout << boost::format("%d points out of %d of scan2 are present in %d voxels of scan1")%totalPointInScan1Voxels%scan2->size()%adjTree->getLeafCount() << endl;
	cout << boost::format("Found %d and %d supervoxels in %f ")%supervoxelClusters.size()%SVMapping.size()%time_spent << endl;

	showTestSuperVoxel(SVMapping, scan1, scan2);

	PointCloud<PointXYZL>::Ptr temoCloud = super.getLabeledCloud();
	PointCloud<PointXYZL>::iterator itr = temoCloud->begin();

	int num(0);
	for (;itr!=temoCloud->end(); ++itr) {
		if ((*itr).label != 0)
			num++;
	}

	cout<<"Num: "<<num << endl;

	LabeledLeafMapT::iterator labItr = labeledLeafMap.begin();
	num = 0;
	for(; labItr != labeledLeafMap.end(); ++labItr) {
		num += (*labItr).first->getSize();
	}
	cout<<"Num: "<<num << endl;
	//	computeSupervoxelScan2Data(SVMapping, scan2);
	//	calculateMutualInformation(SVMapping, scan1, scan2);

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

void
showTestSuperVoxel(SVMap& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2) {

	int SV = programOptions.test;

	typename PointCloudT::Ptr newCloud (new PointCloudT);

	SuperVoxelMappingHelper::SimpleVoxelMap::iterator vxlItr = SVMapping[SV] -> getVoxels() -> begin();
	int scanACounter(0), scanBCounter(0);

	for (; vxlItr != SVMapping[SV] -> getVoxels() ->end(); ++vxlItr) {

		SimpleVoxelMappingHelper::Ptr voxel = (*vxlItr).second;
		typename SimpleVoxelMappingHelper::ScanIndexVectorPtr scanAVector = voxel->getScanAIndices();
		typename SimpleVoxelMappingHelper::ScanIndexVectorPtr scanBVector = voxel->getScanBIndices();

		typename SimpleVoxelMappingHelper::ScanIndexVector::iterator vi = scanAVector -> begin();
		for (; vi != scanAVector -> end(); ++vi, ++scanACounter) {

			PointT p = scan1->at(*vi);

			p.r = 255;
			p.g = 0;
			p.b = 0;

			newCloud->push_back(p);
		}

		vi = scanBVector -> begin();

		for (; vi != scanBVector -> end(); ++vi, ++scanBCounter) {

			PointT p = scan2->at(*vi);

			p.r = 0;
			p.g = 255;
			p.b = 0;

			newCloud->push_back(p);
		}



	}

	showPointCloud(newCloud);
}

void
computeSupervoxelScan2Data(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan) {

	SVMap::iterator svItr = SVMapping.begin();

	// iterate through supervoxels and calculate scan1 data (centroid, rgb, normal)

	for (; svItr!=SVMapping.end(); ++svItr) {

		typename SuperVoxelMappingHelper::Ptr svm = svItr->second;
		typename SuperVoxelMappingHelper::SimpleVoxelMapPtr voxelMap = svm->getVoxels();
		typename SuperVoxelMappingHelper::SimpleVoxelMap::iterator vxItr = voxelMap->begin();

		// Voxel Iteration
		for (;vxItr != voxelMap->end(); ++vxItr) {

			typename SimpleVoxelMappingHelper::Ptr voxelMapping = (*vxItr).second;

			if (voxelMapping -> getScanBIndices()->size() == 0)
				continue;

			PointT centroid;
			unsigned int r,g,b;

			// Point Iteration
			for (typename std::vector<int>::iterator i = voxelMapping -> getScanBIndices()->begin(); i != voxelMapping -> getScanBIndices()->end(); ++i) {

				centroid.x += scan->at(*i).x;
				centroid.y += scan->at(*i).y;
				centroid.z += scan->at(*i).z;

				r += scan->at(*i).r;
				g += scan->at(*i).g;
				b += scan->at(*i).b;

			}

			centroid.x /= voxelMapping -> getScanBIndices()->size();
			centroid.y /= voxelMapping -> getScanBIndices()->size();
			centroid.z /= voxelMapping -> getScanBIndices()->size();

			r /= voxelMapping -> getScanBIndices()->size();
			g /= voxelMapping -> getScanBIndices()->size();
			b /= voxelMapping -> getScanBIndices()->size();

			centroid.r = r;
			centroid.g = g;
			centroid.b = b;

			Eigen::Vector4f scanNormal;
			float curvature;

			computePointNormal(*scan, *voxelMapping -> getScanBIndices(), scanNormal, curvature);
			flipNormalTowardsViewpoint (centroid, 0.0f,0.0f,0.0f, scanNormal);
			scanNormal.normalize();

			PointNormal normal;
			normal.x = centroid.x;
			normal.y = centroid.y;
			normal.z = centroid.z;
			normal.normal_x = scanNormal[0];
			normal.normal_y = scanNormal[1];
			normal.normal_z = scanNormal[2];
			normal.curvature = curvature;
			normal.data_c;

			typename pcl::RGB rgb;
			rgb.r = centroid.r;
			rgb.g = centroid.g;
			rgb.b = centroid.b;
			rgb.a = 255;

			voxelMapping -> setrgbB(rgb);
			voxelMapping -> setNormalB(normal);
		}
	}
}


void
computeSupervoxelScan1Data(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan) {

	SVMap::iterator svItr = SVMapping.begin();

	// iterate through supervoxels and calculate scan1 data (centroid, rgb, normal)

	for (; svItr!=SVMapping.end(); ++svItr) {

		typename SuperVoxelMappingHelper::Ptr svm = svItr->second;
		typename SuperVoxelMappingHelper::SimpleVoxelMapPtr voxelMap = svm->getVoxels();
		typename SuperVoxelMappingHelper::SimpleVoxelMap::iterator vxItr = voxelMap->begin();

		// Voxel Iteration
		for (;vxItr != voxelMap->end(); ++vxItr) {

			typename SimpleVoxelMappingHelper::Ptr voxelMapping = (*vxItr).second;

			if (voxelMapping -> getScanAIndices()->size() == 0)
				continue;

			PointT centroid;
			unsigned int r,g,b;

			// Point Iteration
			for (typename std::vector<int>::iterator i = voxelMapping -> getScanAIndices()->begin(); i != voxelMapping -> getScanAIndices()->end(); ++i) {

				centroid.x += scan->at(*i).x;
				centroid.y += scan->at(*i).y;
				centroid.z += scan->at(*i).z;

				r += scan->at(*i).r;
				g += scan->at(*i).g;
				b += scan->at(*i).b;

			}

			centroid.x /= voxelMapping -> getScanAIndices()->size();
			centroid.y /= voxelMapping -> getScanAIndices()->size();
			centroid.z /= voxelMapping -> getScanAIndices()->size();

			r /= voxelMapping -> getScanAIndices()->size();
			g /= voxelMapping -> getScanAIndices()->size();
			b /= voxelMapping -> getScanAIndices()->size();

			centroid.r = r;
			centroid.g = g;
			centroid.b = b;

			Eigen::Vector4f scanNormal;
			float curvature;

			computePointNormal(*scan, *voxelMapping -> getScanAIndices(), scanNormal, curvature);
			flipNormalTowardsViewpoint (centroid, 0.0f,0.0f,0.0f, scanNormal);
			scanNormal.normalize();

			PointNormal normal;
			normal.x = centroid.x;
			normal.y = centroid.y;
			normal.z = centroid.z;
			normal.normal_x = scanNormal[0];
			normal.normal_y = scanNormal[1];
			normal.normal_z = scanNormal[2];
			normal.curvature = curvature;
			normal.data_c;

			typename pcl::RGB rgb;
			rgb.r = centroid.r;
			rgb.g = centroid.g;
			rgb.b = centroid.b;
			rgb.a = 255;

			voxelMapping -> setrgbA(rgb);
			voxelMapping -> setNormalA(normal);

		}
	}
}


void
calculateMutualInformation(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2) {

	//	map<uint, typename SuperVoxelMappingHelper::Ptr>::iterator svItr = SVMapping.begin();
	//
	//	cout << "" << setw(10) << "Label" << setw(10) << "A" << setw(10) << "B" << setw(10) << "N_Theta" << setw(10) << "Delta_RGB" << endl;
	//
	//	for (; svItr!=SVMapping.end(); ++svItr) {
	//
	//		// Write MI Code
	//
	//		typename SuperVoxelMappingHelper::Ptr svm = svItr->second;
	//
	//		boost::shared_ptr<vector<int> > scanAIndices = svm -> getScanAIndices();
	//		boost::shared_ptr<vector<int> > scanBIndices = svm -> getScanBIndices();
	//
	//		if (scanAIndices->size() < 20 || scanBIndices->size() < 10)
	//			continue;
	//
	//		PointNormal normA = svm->getNormalA();
	//		PointNormal normB = svm->getNormalB();
	//
	//		float theta;
	//
	//		Eigen::Vector3f normalAVector = normA.getNormalVector3fMap();
	//		Eigen::Vector3f normalBVEctor = normB.getNormalVector3fMap();
	//		float dotPro = normalAVector.dot(normalBVEctor);
	//		theta = (180.00 / M_PI) * acos( dotPro/ ( normalAVector.norm() * normalBVEctor.norm()) );
	//		//theta = acos()
	//
	//		Eigen::Vector4f scan2Normal;
	//		float curvature;
	//		computePointNormal(*scan2, *scanBIndices, scan2Normal, curvature);
	//
	//		RGB rgbA = svm->getrgbA();
	//		RGB rgbB = svm->getrgbB();
	//
	//		int r = rgbA.r;
	//		r -= rgbB.r;
	//
	//		int g = rgbA.g;
	//		g -= rgbB.g;
	//
	//		int b = rgbA.b;
	//		b -= rgbB.b;
	//
	//		cout << "" << setw(10) << svItr->first << setw(10) << (svItr->second)->getScanAIndices()->size() << setw(10) << (svItr->second)->getScanBIndices()->size() << setw(10) << theta << setw(10) << boost::format("%d,%d,%d")%r%g%b << endl;
	//	}


}

void
createSuperVoxelMappingForScan2 (SVMap& SVMapping, const typename PointCloudT::Ptr scan, LabeledLeafMapT& labeledLeafMapping, const AdjacencyOctreeT& adjTree) {

	PointCloud<PointT>::iterator scanItr = scan->begin();
	int scanCounter = 0;

	for (;scanItr != scan->end(); ++scanItr, ++scanCounter) {

		PointT a = (*scanItr);

		bool presentInVoxel = adjTree -> isVoxelOccupiedAtPoint(a);

		octree::OctreeKey leafKey;

		genOctreeKeyforPoint(adjTree, a, leafKey);

		MOctreeKey mKey(leafKey);

		if (presentInVoxel) {

			typename SupervoxelClusteringT::LeafContainerT* leaf = adjTree -> getLeafContainerAtPoint(a);

			// check if leaf exists in the mapping from leaf to label
			if (labeledLeafMapping.find(leaf) != labeledLeafMapping.end()) {

				unsigned int label = labeledLeafMapping[leaf];

				// check if SVMapping already contains the supervoxel
				if (SVMapping.find(label) != SVMapping.end()) {

					typename SuperVoxelMappingHelper::SimpleVoxelMapPtr simpleVoxelMapping = SVMapping[label] -> getVoxels();

					// Check if SV contains voxel
					if (simpleVoxelMapping->find(mKey) != simpleVoxelMapping->end()) {
						simpleVoxelMapping->at(mKey)->getScanBIndices()->push_back(scanCounter);
					} else {
						// do nothing if scan1 has no occupied voxel
					}

				} else {
					// do nothing if scan1 has no occupied supervoxel
				}

				// end if

			}

		}
	}

}

void
createSuperVoxelMappingForScan1 (SVMap& SVMapping, const typename PointCloudT::Ptr scan, LabeledLeafMapT& labeledLeafMapping, const AdjacencyOctreeT& adjTree) {

	PointCloud<PointT>::iterator scanItr = scan->begin();
	int scanCounter = 0;

	for (;scanItr != scan->end(); ++scanItr, ++scanCounter) {

		PointT a = (*scanItr);

		bool presentInVoxel = adjTree -> isVoxelOccupiedAtPoint(a);

		octree::OctreeKey leafKey;
		genOctreeKeyforPoint(adjTree, a, leafKey);

		MOctreeKey mKey(leafKey);

		if (presentInVoxel) {

			typename SupervoxelClusteringT::LeafContainerT* leaf = adjTree -> getLeafContainerAtPoint(a);
//			int newLabel = leaf -> getData().owner_->getLabel();

			// check if leaf exists in the mapping from leaf to label
			if (labeledLeafMapping.find(leaf) != labeledLeafMapping.end()) {

				unsigned int label = labeledLeafMapping[leaf];

				// check if SVMapping already contains the supervoxel
				if (SVMapping.find(label) != SVMapping.end()) {

					typename SuperVoxelMappingHelper::SimpleVoxelMapPtr simpleVoxelMapping = SVMapping[label] -> getVoxels();

					// Check if SV contains voxel
					if (simpleVoxelMapping->find(mKey) != simpleVoxelMapping->end()) {
						simpleVoxelMapping->at(mKey)->getScanAIndices()->push_back(scanCounter);
					} else {
						// Create a voxel struct and add to SV
						typename SimpleVoxelMappingHelper::Ptr simpleVoxel (new SimpleVoxelMappingHelper());
						simpleVoxel->getScanAIndices()->push_back(scanCounter);
						simpleVoxelMapping->insert(pair<MOctreeKey, typename SimpleVoxelMappingHelper::Ptr>(mKey, simpleVoxel));
					}

				} else {
					typename SuperVoxelMappingHelper::Ptr newPtr (new SuperVoxelMappingHelper(label));
					typename SimpleVoxelMappingHelper::Ptr simpleVoxel (new SimpleVoxelMappingHelper());
					simpleVoxel->getScanAIndices()->push_back(scanCounter);

					// Add voxel to SV Map
					newPtr->getVoxels()->insert(pair <MOctreeKey, typename SimpleVoxelMappingHelper::Ptr> (mKey, simpleVoxel));

					// Add SV to SVMapping
					SVMapping.insert(pair<uint, typename SuperVoxelMappingHelper::Ptr>(label, newPtr));
				}

				// end if

			}

		} else
			cout << "Not present in voxel"<<endl;
	}

}








