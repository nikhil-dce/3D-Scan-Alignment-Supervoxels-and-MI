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
computeVoxelCentroidScan1(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan, const LabeledLeafMapT& labeledLeafMap);

void
computeVoxelCentroidScan2(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan, const LabeledLeafMapT& labeledLeafMap);

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

	computeVoxelCentroidScan1(SVMapping, scan1, labeledLeafMap);
	computeVoxelCentroidScan2(SVMapping, scan2, labeledLeafMap);

	clock_t end = clock();
	double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

	cout << boost::format("Found %d and %d supervoxels in %f ")%supervoxelClusters.size()%SVMapping.size()%time_spent << endl;

	//	showTestSuperVoxel(SVMapping, scan1, scan2);

	calculateMutualInformation(SVMapping, scan1, scan2);

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
computeVoxelCentroidScan2(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan, const LabeledLeafMapT& labeledLeafMap) {

	SVMap::iterator svItr = SVMapping.begin();
	PointCloudT centroidVoxelCloud;
	int cloudCounter(0);

	// iterate through supervoxels and calculate scan1 data (centroid, rgb, normal)

	for (; svItr!=SVMapping.end(); ++svItr) {

		typename SuperVoxelMappingHelper::Ptr svm = svItr->second;
		typename SuperVoxelMappingHelper::SimpleVoxelMapPtr voxelMap = svm->getVoxels();
		typename SuperVoxelMappingHelper::SimpleVoxelMap::iterator vxItr = voxelMap->begin();

		// Voxel Iteration
		for (;vxItr != voxelMap->end(); ++vxItr, ++cloudCounter) {

			typename SimpleVoxelMappingHelper::Ptr voxelMapping = (*vxItr).second;

			PointT centroid;
			unsigned int r,g,b;

			if (voxelMapping->getScanBIndices()->size() != 0) {

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
				centroid.a = 255;


			}

			centroidVoxelCloud.push_back(centroid);
			voxelMapping -> setIndexB(cloudCounter); // index will be same for both scans
			voxelMapping -> setCentroidB(centroid);
		}
	}

	// Iterate again for normals
	svItr = SVMapping.begin();

	for (; svItr!=SVMapping.end(); ++svItr) {

		typename SuperVoxelMappingHelper::Ptr svm = svItr->second;
		typename SuperVoxelMappingHelper::SimpleVoxelMapPtr voxelMap = svm->getVoxels();
		typename SuperVoxelMappingHelper::SimpleVoxelMap::iterator vxItr = voxelMap->begin();

		// Voxel Iteration
		for (;vxItr != voxelMap->end(); ++vxItr) {

			SupervoxelClusteringT::LeafContainerT* leaf = (*vxItr).first;
			typename SimpleVoxelMappingHelper::Ptr voxel = (*vxItr).second;
			vector<int> indicesForNormal;

			if (voxel->getScanBIndices()->size() == 0) {
				voxel->setNormalB(PointNormal());
				continue;
			}

			indicesForNormal.push_back(voxel->getIndexB());

			typename SupervoxelClusteringT::LeafContainerT::const_iterator leafNItr = leaf->cbegin();
			for (; leafNItr != leaf->cend(); ++leafNItr) {

				SupervoxelClusteringT::LeafContainerT* neighborLeaf = (*leafNItr);

				if (voxelMap -> find(neighborLeaf) != voxelMap->end()) {
					SimpleVoxelMappingHelper::Ptr neighborSimpleVoxel = voxelMap->at(neighborLeaf);

					if (neighborSimpleVoxel->getScanBIndices()->size() == 0)
						continue;

					indicesForNormal.push_back(neighborSimpleVoxel->getIndexB());
				}
			}


			// Normal Call

			Eigen::Vector4f voxelNormal;
			float curvature;
			PointT centroid = voxel->getCentroidB();

			if (indicesForNormal.size() > 3) {
				computePointNormal(centroidVoxelCloud, indicesForNormal, voxelNormal, curvature);
				flipNormalTowardsViewpoint (centroid, 0.0f,0.0f,0.0f, voxelNormal);

				voxelNormal[3] = 0.0f;
				voxelNormal.normalize();

			}

			PointNormal normal;
			normal.x = centroid.x;
			normal.y = centroid.y;
			normal.z = centroid.z;
			normal.normal_x = voxelNormal[0];
			normal.normal_y = voxelNormal[1];
			normal.normal_z = voxelNormal[2];
			normal.curvature = curvature;
			normal.data_c;

			voxel->setNormalB(normal);
		}
	}

}

void
computeVoxelCentroidScan1(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan, const LabeledLeafMapT& labeledLeafMap) {

	SVMap::iterator svItr = SVMapping.begin();
	PointCloudT centroidVoxelCloud;
	int cloudCounter(0);

	// iterate through supervoxels and calculate scan1 data (centroid, rgb, normal)

	for (; svItr!=SVMapping.end(); ++svItr) {

		typename SuperVoxelMappingHelper::Ptr svm = svItr->second;
		typename SuperVoxelMappingHelper::SimpleVoxelMapPtr voxelMap = svm->getVoxels();
		typename SuperVoxelMappingHelper::SimpleVoxelMap::iterator vxItr = voxelMap->begin();

		// Voxel Iteration
		for (;vxItr != voxelMap->end(); ++vxItr, ++cloudCounter) {

			typename SimpleVoxelMappingHelper::Ptr voxel = (*vxItr).second;

			PointT centroid;
			unsigned int r,g,b;

			if (voxel->getScanAIndices()->size() != 0) {

				// Point Iteration
				for (typename std::vector<int>::iterator i = voxel -> getScanAIndices()->begin(); i != voxel -> getScanAIndices()->end(); ++i) {

					centroid.x += scan->at(*i).x;
					centroid.y += scan->at(*i).y;
					centroid.z += scan->at(*i).z;

					r += scan->at(*i).r;
					g += scan->at(*i).g;
					b += scan->at(*i).b;

				}

				centroid.x /= voxel -> getScanAIndices()->size();
				centroid.y /= voxel -> getScanAIndices()->size();
				centroid.z /= voxel -> getScanAIndices()->size();

				r /= voxel -> getScanAIndices()->size();
				g /= voxel -> getScanAIndices()->size();
				b /= voxel -> getScanAIndices()->size();

				centroid.r = r;
				centroid.g = g;
				centroid.b = b;
				centroid.a = 255;
			}

			centroidVoxelCloud.push_back(centroid);
			voxel -> setIndexA(cloudCounter); // index will be same for both scans
			voxel -> setCentroidA(centroid);
		}
	}

	// Iterate again for normals
	svItr = SVMapping.begin();

	for (; svItr!=SVMapping.end(); ++svItr) {

		typename SuperVoxelMappingHelper::Ptr svm = svItr->second;
		typename SuperVoxelMappingHelper::SimpleVoxelMapPtr voxelMap = svm->getVoxels();
		typename SuperVoxelMappingHelper::SimpleVoxelMap::iterator vxItr = voxelMap->begin();

		// Voxel Iteration
		for (;vxItr != voxelMap->end(); ++vxItr) {

			SupervoxelClusteringT::LeafContainerT* leaf = (*vxItr).first;
			typename SimpleVoxelMappingHelper::Ptr voxel = (*vxItr).second;

			if (voxel -> getIndexA() == 0) {
				voxel->setNormalA(PointNormal());
				continue;
			}

			vector<int> indicesForNormal;

			indicesForNormal.push_back(voxel->getIndexA());

			typename SupervoxelClusteringT::LeafContainerT::const_iterator leafNItr = leaf->cbegin();
			for (; leafNItr != leaf->cend(); ++leafNItr) {

				SupervoxelClusteringT::LeafContainerT* neighborLeaf = (*leafNItr);

				if (voxelMap->find(neighborLeaf) != voxelMap->end()) {
					SimpleVoxelMappingHelper::Ptr neighborSimpleVoxel = voxelMap->at(neighborLeaf);

					if (neighborSimpleVoxel->getScanAIndices()->size() == 0)
						continue;

					indicesForNormal.push_back(neighborSimpleVoxel->getIndexA());
				}
			}

			// Normal Call

			Eigen::Vector4f voxelNormal = Eigen::Vector4f::Zero();
			float curvature;
			PointT centroid = voxel->getCentroidA();

			if (indicesForNormal.size() > 3) {
				computePointNormal(centroidVoxelCloud, indicesForNormal, voxelNormal, curvature);
				flipNormalTowardsViewpoint (centroid, 0.0f,0.0f,0.0f, voxelNormal);

				voxelNormal[3] = 0.0f;
				voxelNormal.normalize();

			}

			PointNormal normal;
			normal.x = centroid.x;
			normal.y = centroid.y;
			normal.z = centroid.z;
			normal.normal_x = voxelNormal[0];
			normal.normal_y = voxelNormal[1];
			normal.normal_z = voxelNormal[2];
			normal.curvature = curvature;
			normal.data_c;

			voxel->setNormalA(normal);
		}

	}

}

void
calculateMutualInformation(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2) {

	map<uint, typename SuperVoxelMappingHelper::Ptr>::iterator svItr = SVMapping.begin();

	cout << "" << setw(10) << "Label" << setw(10) << "A" << setw(10) << "B" << setw(10) << "N_Theta" << setw(10) << "Delta_RGB" << endl;

	for (; svItr!=SVMapping.end(); ++svItr) {

		// Write MI Code

		int svLabel = svItr->first;
		bool debug = false;

		typename SuperVoxelMappingHelper::Ptr svm = svItr->second;

		SuperVoxelMappingHelper::SimpleVoxelMapPtr vxlMapPtr = svm->getVoxels();
		SuperVoxelMappingHelper::SimpleVoxelMap::iterator vxlMapItr = vxlMapPtr->begin();

		Eigen::Vector3f svNormA = Eigen::Vector3f::Zero();
		Eigen::Vector3f svNormB = Eigen::Vector3f::Zero();

		int counterA(0), counterB(0);

		for (; vxlMapItr != vxlMapPtr -> end(); ++ vxlMapItr) {

			SimpleVoxelMappingHelper::Ptr voxel = (*vxlMapItr).second;

			PointNormal normA = voxel->getNormalA();
			PointNormal normB = voxel->getNormalB();

			//			if (debug) {
			//				cout<<boost::format("%d A: %d %f %f %f")%svItr->first%voxel->getScanAIndices()->size()%normA.normal_x%normA.normal_y%normA.normal_z<<endl;
			//				cout<<boost::format("%d B: %d %f %f %f")%svItr->first%voxel->getScanBIndices()->size()%normB.normal_x%normB.normal_y%normB.normal_z<<endl;
			//			}

			counterA += voxel->getScanAIndices()->size();
			counterB += voxel->getScanBIndices()->size();

			svNormA += normA.getNormalVector3fMap();
			svNormB += normB.getNormalVector3fMap();
		}


		double theta;
		double dotPro = svNormA.dot(svNormB);
		double norm = (svNormA.norm() * svNormB.norm());
		theta = (180.00 / M_PI) * acos(dotPro/norm);

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

		if (counterA > 5 && counterB > 5) {

			float normX = svNormA[0];
			float normY = svNormA[1];
			float normZ = svNormA[2];
			cout<<boost::format("%d A: %d %f %f %f")%svItr->first%counterA%normX%normY%normZ<<endl;

			normX = svNormB[0];
			normY = svNormB[1];
			normZ = svNormB[2];

			cout<<boost::format("%d B: %d %f %f %f")%svItr->first%counterB%normX%normY%normZ<<endl;

			cout << svItr->first <<" Theta: "<< theta << endl;
		}

		//		cout << "" << setw(10) << svItr->first << setw(10) << (svItr->second)->getScanAIndices()->size() << setw(10) << (svItr->second)->getScanBIndices()->size() << setw(10) << theta << setw(10) << boost::format("%d,%d,%d")%r%g%b << endl;
	}


}

void
createSuperVoxelMappingForScan2 (SVMap& SVMapping, const typename PointCloudT::Ptr scan, LabeledLeafMapT& labeledLeafMapping, const AdjacencyOctreeT& adjTree) {

	PointCloud<PointT>::iterator scanItr = scan->begin();
	int scanCounter = 0;

	for (;scanItr != scan->end(); ++scanItr, ++scanCounter) {

		PointT a = (*scanItr);

		bool presentInVoxel = adjTree -> isVoxelOccupiedAtPoint(a);

		if (presentInVoxel) {

			typename SupervoxelClusteringT::LeafContainerT* leaf = adjTree -> getLeafContainerAtPoint(a);

			// check if leaf exists in the mapping from leaf to label
			if (labeledLeafMapping.find(leaf) != labeledLeafMapping.end()) {

				unsigned int label = labeledLeafMapping[leaf];

				// check if SVMapping already contains the supervoxel
				if (SVMapping.find(label) != SVMapping.end()) {

					typename SuperVoxelMappingHelper::SimpleVoxelMapPtr simpleVoxelMapping = SVMapping[label] -> getVoxels();

					// Check if SV contains voxel
					if (simpleVoxelMapping->find(leaf) != simpleVoxelMapping->end()) {
						simpleVoxelMapping->at(leaf)->getScanBIndices()->push_back(scanCounter);
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

		if (presentInVoxel) {

			typename SupervoxelClusteringT::LeafContainerT* leaf = adjTree -> getLeafContainerAtPoint(a);

			// check if leaf exists in the mapping from leaf to label
			if (labeledLeafMapping.find(leaf) != labeledLeafMapping.end()) {

				unsigned int label = labeledLeafMapping[leaf];

				// check if SVMapping already contains the supervoxel
				if (SVMapping.find(label) != SVMapping.end()) {

					typename SuperVoxelMappingHelper::SimpleVoxelMapPtr simpleVoxelMapping = SVMapping[label] -> getVoxels();


					if (simpleVoxelMapping->find(leaf) != simpleVoxelMapping->end()) {
						simpleVoxelMapping->at(leaf)->getScanAIndices()->push_back(scanCounter);
					} else {
						// Create a voxel struct and add to SV
						typename SimpleVoxelMappingHelper::Ptr simpleVoxel (new SimpleVoxelMappingHelper());
						simpleVoxel->getScanAIndices()->push_back(scanCounter);
						simpleVoxelMapping->insert(pair<SupervoxelClusteringT::LeafContainerT*, typename SimpleVoxelMappingHelper::Ptr>(leaf, simpleVoxel));
					}

				} else {
					typename SuperVoxelMappingHelper::Ptr newPtr (new SuperVoxelMappingHelper(label));
					typename SimpleVoxelMappingHelper::Ptr simpleVoxel (new SimpleVoxelMappingHelper());
					simpleVoxel->getScanAIndices()->push_back(scanCounter);

					// Add voxel to SV Map
					newPtr->getVoxels()->insert(pair <SupervoxelClusteringT::LeafContainerT*, typename SimpleVoxelMappingHelper::Ptr> (leaf, simpleVoxel));

					// Add SV to SVMapping
					SVMapping.insert(pair<uint, typename SuperVoxelMappingHelper::Ptr>(label, newPtr));
				}

				// end if

			}

		} else
			cout << "Not present in voxel"<<endl;
	}

}








