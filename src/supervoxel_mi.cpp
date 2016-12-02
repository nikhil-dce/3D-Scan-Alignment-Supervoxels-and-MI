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


#include <gsl/gsl_multimin.h>

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

// Should be a factor of 1.0
#define NORM_DX 0.2
#define NORM_DY 0.2
#define NORM_DZ 0.2

struct options {

	float vr;
	float sr;
	float colorWeight;
	float spatialWeight;
	float normalWeight;
	int test;
	bool showScans;

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
showPointClouds(PointCloudT::Ptr, PointCloudT::Ptr, string viewerTitle);

void
showTestSuperVoxel(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2);

void
computeVoxelCentroidScan1(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan, const LabeledLeafMapT& labeledLeafMap);

void
computeVoxelCentroidScan2(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan, const LabeledLeafMapT& labeledLeafMap);

double
calculateMutualInformation(SVMap& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2);

void
createSuperVoxelMappingForScan1 (SVMap& SVMapping, const typename PointCloudT::Ptr scan, LabeledLeafMapT& labeledLeafMapping, const AdjacencyOctreeT& adjTree);

void
createSuperVoxelMappingForScan2 (SVMap& SVMapping, const typename PointCloudT::Ptr scan, LabeledLeafMapT& labeledLeafMapping, const AdjacencyOctreeT& adjTree);

int optimize(SVMap& SVMapping, LabeledLeafMapT& labeledLeafMap, AdjacencyOctreeT& adjTree, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2, gsl_vector* baseX);

int getNormalVectorCode(Eigen::Vector3f vector);

void transform_get_translation(Eigen::Matrix4f t, double *x, double *y, double *z) {

	*x = t(0,3);
	*y = t(1,3);
	*z = t(2,3);

}

void transform_get_rotation(Eigen::Matrix4f t, double *x, double *y, double *z) {

	double a = t(2,1);
	double b = t(2,2);
	double c = t(2,0);
	double d = t(1,0);
	double e = t(0,0);

	*x = atan2(a, b);
	*y = asin(-c);
	*z = atan2(d, e);

}

int initOptions(int argc, char* argv[]) {

	namespace po = boost::program_options;

	programOptions.colorWeight = 0.0f;
	programOptions.spatialWeight = 0.4f;
	programOptions.normalWeight = 1.0f;
	programOptions.vr = 1.0f;
	programOptions.sr = 5.0f;
	programOptions.test = 0; // 324
	programOptions.showScans = false;

	po::options_description desc ("Allowed Options");

	desc.add_options()
							("help,h", "Usage <Scan 1 Path> <Scan 2 Path> <Transform File>")
							("voxel_res,v", po::value<float>(&programOptions.vr), "voxel resolution")
							("seed_res,s", po::value<float>(&programOptions.sr), "seed resolution")
							("color_weight,c", po::value<float>(&programOptions.colorWeight), "color weight")
							("spatial_weight,z", po::value<float>(&programOptions.spatialWeight), "spatial weight")
							("normal_weight,n", po::value<float>(&programOptions.normalWeight), "normal weight")
							("test,t", po::value<int>(&programOptions.test), "test")
							("show_scan,y", po::value<bool>(&programOptions.showScans), "Show scans");

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

	gsl_vector *base_pose;

	if (io::loadPCDFile<PointT> (ss.str(), *scan2)) {
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
		//				Eigen::Affine3f transform = Eigen::Affine3f::Identity();
		string line;

		for (int i = 0; i < 4; i++) {
			std::getline(in,line);

			std::istringstream sin(line);
			for (int j = 0; j < 4; j++) {
				sin >> transform (i,j);
			}
		}

		if (!programOptions.showScans) {
			double x, y, z, roll, pitch, yaw;
			transform_get_translation(transform, &x, &y, &z);
			transform_get_rotation(transform, &roll, &pitch, &yaw);

			base_pose = gsl_vector_alloc (6);
			gsl_vector_set (base_pose, 0, x);
			gsl_vector_set (base_pose, 1, y);
			gsl_vector_set (base_pose, 2, z);
			gsl_vector_set (base_pose, 3, roll);
			gsl_vector_set (base_pose, 4, pitch);
			gsl_vector_set (base_pose, 5, yaw);
		} else {

			transformPointCloud (*scan2, *temp, (Eigen::Matrix4f) transform.inverse());
			scan2->clear();

		}
	}

	if (programOptions.showScans && programOptions.test == 0) {
		showPointClouds(scan1, temp, "Supervoxel Based MI Viewer: " + transformFile);
		return 0;
	}

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

	LabeledLeafMapT labeledLeafMap;
	super.getLabeledLeafContainerMap(labeledLeafMap);

	pcl::SupervoxelClustering<PointT>::OctreeAdjacencyT::Ptr adjTree = super.getOctreeeAdjacency();

	cout << "LeafCount " << adjTree->getLeafCount() << ' ' << labeledLeafMap.size() << endl;

	SVMap SVMapping;
	createSuperVoxelMappingForScan1(SVMapping,scan1, labeledLeafMap, adjTree);

	if (programOptions.test && programOptions.test != 0) {
		createSuperVoxelMappingForScan1(SVMapping, temp, labeledLeafMap, adjTree);
		showTestSuperVoxel(SVMapping, scan1, temp);
		return 0;
	}

	computeVoxelCentroidScan1(SVMapping, scan1, labeledLeafMap);


	optimize(SVMapping, labeledLeafMap, adjTree, scan1, scan2, base_pose);

	clock_t end = clock();
	double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
	cout << boost::format("Found %d and %d supervoxels in %f ")%supervoxelClusters.size()%SVMapping.size()%time_spent << endl;

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

void
computeVoxelCentroidScan2(SVMap& SVMapping, PointCloudT::Ptr scan, const LabeledLeafMapT& labeledLeafMap) {

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

		int label = (*svItr).first;
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
				// Not needed as we have to compare normal of sv for directions
				//				flipNormalTowardsViewpoint (centroid, 0.0f,0.0f,0.0f, voxelNormal);

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

			if (voxel->getScanAIndices()->size() == 0) {
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
				//				flipNormalTowardsViewpoint (centroid, 0.0f,0.0f,0.0f, voxelNormal);

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

#define MIN_POINTS_IN_SUPERVOXEL 5

double
calculateMutualInformation(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2) {

	//	cout << "Calculating Multual Information... " << endl;
	SVMap::iterator svItr = SVMapping.begin();

	double mi;

	map<int, int> randomX;
	map<int, int> randomY;
	map<string, int> randomXY;

	int size(0);

	for (; svItr!=SVMapping.end(); ++svItr) {

		// Write MI Code
		int svLabel = svItr->first;
		typename SuperVoxelMappingHelper::Ptr supervoxel = svItr->second;

		SuperVoxelMappingHelper::SimpleVoxelMapPtr voxelMap = supervoxel->getVoxels();
		SuperVoxelMappingHelper::SimpleVoxelMap::iterator voxelItr = voxelMap->begin();

		Eigen::Vector3f svNormA = Eigen::Vector3f::Zero();
		Eigen::Vector3f svNormB = Eigen::Vector3f::Zero();

		int counterA(0), counterB(0);

		//		cout << "Computing Supervoxel Normal " << supervoxel->getVoxels()->size() << endl;
		for (; voxelItr != voxelMap -> end(); ++ voxelItr) {

			SimpleVoxelMappingHelper::Ptr voxel = (*voxelItr).second;

			PointNormal normA = voxel->getNormalA();
			PointNormal normB = voxel->getNormalB();

			counterA += voxel->getScanAIndices()->size();
			counterB += voxel->getScanBIndices()->size();

			svNormA += normA.getNormalVector3fMap();
			svNormB += normB.getNormalVector3fMap();
		}


		if (counterA > MIN_POINTS_IN_SUPERVOXEL && counterB > MIN_POINTS_IN_SUPERVOXEL) {

			size++;

			svNormA.normalize();
			svNormB.normalize();

			// cache A code
			int codeA = getNormalVectorCode(svNormA);
			int codeB = getNormalVectorCode(svNormB);


			if (randomX.find(codeA) != randomX.end()) {
				randomX[codeA]++;
			}  else {
				randomX.insert(pair<int, int> (codeA, 1));
			}

			if (randomY.find(codeB) != randomY.end())
				randomY[codeB]++;
			else
				randomY.insert(pair<int, int> (codeB, 1));

			string codePair = boost::str(boost::format("%d_%d")%codeA%codeB);

			if (randomXY.find(codePair) != randomXY.end())
				randomXY[codePair]++;
			else
				randomXY.insert(pair<string, int>(codePair, 1));

			//			double theta;
			//			double dotPro = svNormA.dot(svNormB);
			//			theta = (180.00 / M_PI) * acos(dotPro);
			//
			//			float normX = svNormA[0];
			//			float normY = svNormA[1];
			//			float normZ = svNormA[2];
			//			cout<<boost::format("%d A: %d %f %f %f")%svItr->first%counterA%normX%normY%normZ<<endl;
			//
			//			normX = svNormB[0];
			//			normY = svNormB[1];
			//			normZ = svNormB[2];
			//
			//			cout<<boost::format("%d B: %d %f %f %f")%svItr->first%counterB%normX%normY%normZ<<endl;
			//
			//			cout << svItr->first <<" Theta: "<< theta << endl;
		}

	}

	// calculate MI on randomX and randomY

	// calculating H(X)
	map<int, int>::iterator itr;
	double hX = 0;

	for (itr = randomX.begin(); itr != randomX.end(); ++itr) {
		double x = ((double)itr->second) / size;
		hX += x * log(x);
	}

	hX *= -1;

	// calculating H(Y)
	double hY = 0;

	for (itr = randomY.begin(); itr != randomY.end(); ++itr) {
		double y = ((double)itr->second) / size;
		hY += y * log(y);
	}

	hY *= -1;

	// calculating H(X,Y)
	map<string, int>::iterator xyItr;
	double hXY = 0;

	for (xyItr = randomXY.begin(); xyItr != randomXY.end(); ++xyItr) {
		double xy = ((double)xyItr->second) / size;
		hXY += xy * log(xy);
	}

	hXY *= -1;

	mi = hX + hY - hXY;

	return mi;
}

void
createSuperVoxelMappingForScan2 (SVMap& SVMapping, const typename PointCloudT::Ptr scan, LabeledLeafMapT& labeledLeafMapping, const AdjacencyOctreeT& adjTree) {

	PointCloud<PointT>::iterator scanItr = scan->begin();
	int scanCounter = 0;
	int totalPresentInVoxel = 0;

	for (;scanItr != scan->end(); ++scanItr, ++scanCounter) {

		PointT a = (*scanItr);

		bool presentInVoxel = adjTree -> isVoxelOccupiedAtPoint(a);

		if (presentInVoxel) {

			totalPresentInVoxel++;
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

	//	cout << "Scan 2 points present in voxels: " << totalPresentInVoxel << endl;
}

struct MI_Opti_Data{

	SVMap* svMap;
	LabeledLeafMapT* labeledLeafMap;
	PointCloudT::Ptr scan1;
	PointCloudT::Ptr scan2;
	AdjacencyOctreeT* adjTree;
};

/*
 *	Optimization Function
 * 	Note: Scan A Data remains constant including the supervoxels generated
 * 	v is the transformation vector XYZRPY best guess till now
 * 	params will include :-
 *
 * 	1. SVMap -> Supervoxel to scanA Indices will remain constant
 * 	2. SVMap -> Supervoxel to scanB Indices (will change in every iteration)
 * 	3. LeafMapContainer -> This will always be constant (Will be used to find supervoxels for scan B points)
 * 	4. Octree -> To find the leaf container for scan B points
 * 	5. Scan A
 * 	6. Scan B
 *
 *	Function steps:
 *
 *	1. Transform B with current XYZRPY
 *	2. Find the corresponding Supervoxels
 *	3. Find the supervoxels with a minimum number of points
 *	4. Apply Mutual Information on the common Supervoxels
 *
 *	CommonSupervoxels will be used for all the steps below
 *
 *	Mutual Information Steps (Only Normal Feature for now):
 *
 *	A. Feature space -> Normal
 *
 *		1. Find The normals for scanB in the supervoxels
 *		2. Find Normal Vector Code for scan B in all supervoxels
 *		3. Calculate H(X) for the supervoxels on the basis of Normal Vector Codes for Scan A
 *		4. Calculate H(Y) for the supervoxels on the basis of Normal Vector Codes for Scan B
 *		5. Calculate H(X, Y) for the supervoxels on the basis of Normal Vector Codes for both Scan A and Scan B (Joint Histo)
 *		6. Calculate MI(X,Y) = H(X) + H (Y) - H(X,Y)
 *		7. return as the value of the function -MI(X,Y)
 *
 */

double mi_f (const gsl_vector *pose, void* params) {

	// Initialize All Data
	double x, y, z, roll, pitch ,yaw;
	x = gsl_vector_get(pose, 0);
	y = gsl_vector_get(pose, 1);
	z = gsl_vector_get(pose, 2);
	roll = gsl_vector_get(pose, 3);
	pitch = gsl_vector_get(pose, 4);
	yaw = gsl_vector_get(pose, 5);

	MI_Opti_Data* miOptiData = (MI_Opti_Data*) params;

	PointCloudT::Ptr scan1 = miOptiData->scan1;
	PointCloudT::Ptr scan2 = miOptiData->scan2;
	PointCloudT::Ptr transformedScan2 =  boost::shared_ptr<PointCloudT>(new PointCloudT());

	SVMap* SVMapping = miOptiData->svMap;
	LabeledLeafMapT* labeledLeafMap = miOptiData->labeledLeafMap;
	AdjacencyOctreeT* adjTree = miOptiData->adjTree;

	// Create Transformation
	Eigen::Affine3f transform = Eigen::Affine3f::Identity();
	transform.translation() << x,y,z;
	transform.rotate (Eigen::AngleAxisf (roll, Eigen::Vector3f::UnitX()));
	transform.rotate (Eigen::AngleAxisf (pitch, Eigen::Vector3f::UnitY()));
	transform.rotate(Eigen::AngleAxisf (yaw, Eigen::Vector3f::UnitZ()));

	// Transform point cloud
	pcl::transformPointCloud(*scan2, *transformedScan2, transform);

	// Clear SVMap for new scan2 properties
	SVMap::iterator svItr = SVMapping->begin();

	for (; svItr != SVMapping->end(); ++svItr) {
		int label = svItr->first;
		SuperVoxelMappingHelper::Ptr svMapHelper = svItr->second;

		typename SuperVoxelMappingHelper::SimpleVoxelMapPtr voxelMap = svMapHelper->getVoxels();
		typename SuperVoxelMappingHelper::SimpleVoxelMap::iterator voxelItr = voxelMap->begin();

		for (; voxelItr != voxelMap->end(); ++voxelItr) {

			typename SupervoxelClusteringT::LeafContainerT* leaf = voxelItr->first;
			SimpleVoxelMappingHelper::Ptr voxel = voxelItr->second;

			voxel->clearScanBData();
		}
	}

	// recreate map for scan2
	createSuperVoxelMappingForScan2(*SVMapping, transformedScan2, *labeledLeafMap, *adjTree);

	// compute Voxel Data for scan 2
	computeVoxelCentroidScan2(*SVMapping, transformedScan2, *labeledLeafMap);


	double mi = calculateMutualInformation(*SVMapping, scan1, transformedScan2);

	return -mi;
}

int optimize(SVMap& SVMapping, LabeledLeafMapT& labeledLeafMap, AdjacencyOctreeT& adjTree, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2, gsl_vector* baseX) {

	MI_Opti_Data* mod = new MI_Opti_Data();
	mod->adjTree = &adjTree;
	mod->labeledLeafMap = &labeledLeafMap;
	mod->scan1 = scan1;
	mod->scan2 = scan2;
	mod->svMap = &SVMapping;

	const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
	gsl_multimin_fminimizer *s = NULL;

	gsl_vector *ss;
	gsl_multimin_function minex_func;

	size_t iter = 0;
	int status;
	double size;

	/* Set  initial step sizes to 1 */
	ss = gsl_vector_alloc (6);
	gsl_vector_set (ss, 0, 1.0);
	gsl_vector_set (ss, 1, 1.0);
	gsl_vector_set (ss, 2, 1.0);
	gsl_vector_set (ss, 3, 0.2);
	gsl_vector_set (ss, 4, 0.2);
	gsl_vector_set (ss, 5, 0.2);

	/* Initialize method and iterate */
	minex_func.n = 6; // Dimension
	minex_func.f = mi_f;
	minex_func.params = mod;

	s = gsl_multimin_fminimizer_alloc (T, 6);
	gsl_multimin_fminimizer_set (s, &minex_func, baseX, ss);

	do {

		iter++;
		status = gsl_multimin_fminimizer_iterate(s);

		if (status)
			break;

		size = gsl_multimin_fminimizer_size (s);
		status = gsl_multimin_test_size (size, 1e-2);

		cout << "Iterations: " << iter << endl;

		printf("%5d f() = %7.3f size = %.3f\n",
				iter,
				s->fval,
				size);

		if (status == GSL_SUCCESS) {

			cout << "Base Transformation: " << endl;

			double tx = gsl_vector_get (baseX, 0);
			double ty = gsl_vector_get (baseX, 1);
			double tz = gsl_vector_get (baseX, 2);
			double roll = gsl_vector_get (baseX, 3);
			double pitch = gsl_vector_get (baseX, 4);
			double yaw = gsl_vector_get (baseX, 5);

			cout << "Tx: " << tx << endl;
			cout << "Ty: " << ty << endl;
			cout << "Tz: " << tz << endl;
			cout << "Roll: " << roll << endl;
			cout << "Pitch: " << pitch << endl;
			cout << "Yaw: " << yaw << endl;


			cout << "Converged to minimum at " << endl;

			tx = gsl_vector_get (s->x, 0);
			ty = gsl_vector_get (s->x, 1);
			tz = gsl_vector_get (s->x, 2);
			roll = gsl_vector_get (s->x, 3);
			pitch = gsl_vector_get (s->x, 4);
			yaw = gsl_vector_get (s->x, 5);

			cout << "Tx: " << tx << endl;
			cout << "Ty: " << ty << endl;
			cout << "Tz: " << tz << endl;
			cout << "Roll: " << roll << endl;
			cout << "Pitch: " << pitch << endl;
			cout << "Yaw: " << yaw << endl;

			Eigen::Affine3f resultantTransform;
			resultantTransform.translation() << tx, ty, tz;
			resultantTransform.rotate (Eigen::AngleAxisf (roll, Eigen::Vector3f::UnitX()));
			resultantTransform.rotate (Eigen::AngleAxisf (pitch, Eigen::Vector3f::UnitY()));
			resultantTransform.rotate(Eigen::AngleAxisf (yaw, Eigen::Vector3f::UnitZ()));

			cout << "Resulting Transformation: " << endl << resultantTransform.matrix() << endl;
		}

	} while(status == GSL_CONTINUE && iter < 100);

	//	gsl_vector_free(baseX);
	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free(s);

	return status;
}

/*
 * 	Returns a unique Normal Vector code for a group of normalized vectors in the range
 * 	[x,y,z] - [x+dx, y+dy, z+dy]
 *
 * 	NormalVectorCode is merely a qualitative measure
 * 	It shouldn't be used for comparison/computation
 *
 */
int getNormalVectorCode(Eigen::Vector3f vector) {

	float x = vector[0];
	float y = vector[1];
	float z = vector[2];

	if (x > 1 || y > 1 || z > 1)
		return 0;

	int a(0), b(0), c(0);
	float dx(NORM_DX), dy(NORM_DY), dz(NORM_DZ);

	int Dx(0),Dy(0),Dz(0);
	int Tx(0), Ty(0), Tz(0);

	Dx = 1.0 / dx;
	Dy = 1.0 / dy;
	Dz = 1.0 / dz;

	Tx = 2*Dx + 1;
	Ty = 2*Dy + 1;
	Tz = 2*Dz + 1;

	while (dx < abs(x)) {
		dx += NORM_DX;
		a++;
	}

	while (dy < abs(y)) {
		dy += NORM_DY;
		b++;
	}

	while (dz < abs(z)) {
		dz += NORM_DZ;
		c++;
	}

	if (x >= 0)
		a = Dx + a;
	else
		a = Dx - a;

	if (y >= 0)
		b = Dy + b;
	else
		b = Dy - b;

	if (z >= 0)
		c = Dz + c;
	else
		c = Dx - c;

	int code = a + b * Tx + c * (Tx * Ty);

	return code;
}

void
showPointClouds(PointCloudT::Ptr scan1, PointCloudT::Ptr scan2, string viewerTitle) {

	boost::shared_ptr<visualization::PCLVisualizer> viewer (new visualization::PCLVisualizer (viewerTitle));
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

