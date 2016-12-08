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

#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl/segmentation/supervoxel_clustering.h>

#include "supervoxel_mapping.hpp"
#include <cmath>
#include <iomanip>


#include <gsl/gsl_multimin.h>
#include "supervoxel_octree_pointcloud_adjacency.h"
#include <cmath>

using namespace pcl;
using namespace std;

typedef PointXYZRGBA PointT;
typedef PointCloud<PointT> PointCloudT;
typedef SupervoxelClustering<PointT> SupervoxelClusteringT;
typedef PointXYZL PointLT;
typedef PointCloud<PointLT> PointLCloudT;
typedef KdTreeFLANN<PointXYZ> KdTreeXYZ;
typedef PointCloud<PointXYZ> PointCloudXYZ;
//typedef typename std::vector<typename SuperVoxelMapping::Ptr> SVMappingVector;

// Replace map with boost::unordered_map for faster lookup
typedef std::map<uint, typename SData::Ptr> SVMap;
typedef std::map<typename SupervoxelClusteringT::LeafContainerT*, uint32_t> LabeledLeafMapT;
typedef std::map<typename SupervoxelClusteringT::LeafContainerT*, typename VData::Ptr> LeafVoxelMapT;
typedef typename SupervoxelClusteringT::OctreeAdjacencyT::Ptr AdjacencyOctreeT;

#define SEARCH_SUPERVOXEL_NN 10;

#define NORM_R 5 // 5 meters

// Should be a factor of 1.0
#define NORM_DX 0.1
#define NORM_DY 0.1
#define NORM_DZ 0.1

// Min points to be present in supevoxel for MI consideration
#define MIN_POINTS_IN_SUPERVOXEL 5

struct options {

	float vr;
	float sr;
	float colorWeight;
	float spatialWeight;
	float normalWeight;
	int test;
	bool showScans;

}programOptions;

struct octBounds {
	PointT minPt, maxPt;
}octreeBounds;

template <typename Type>
inline Type square(Type x)
{
	return x * x; // will only work on types for which there is the * operator.
}

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

map <uint32_t, Supervoxel<PointT>::Ptr>
initializeVoxels(
		PointCloudT::Ptr scan1,
		PointCloudT::Ptr scan2,
		SupervoxelClusteringT& super,
		SVMap& supervoxelMapping,
		LabeledLeafMapT& labeledLeafMap);

void
showPointCloud(typename PointCloudT::Ptr);

void
showPointClouds(PointCloudT::Ptr, PointCloudT::Ptr, string viewerTitle);

void
showTestSuperVoxel(map<uint, typename SuperVoxelMappingHelper::Ptr>& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2);

double
calculateMutualInformation(SVMap& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2);

void
createSuperVoxelMappingForScan1 (SVMap& SVMapping, const typename PointCloudT::Ptr scan, LabeledLeafMapT& labeledLeafMapping, const AdjacencyOctreeT& adjTree);

void
createSuperVoxelMappingForScan2 (
		SVMap& SVMapping,
		const typename PointCloudT::Ptr scan,
		LabeledLeafMapT& labeledLeafMapping,
		const AdjacencyOctreeT& adjTree,
		KdTreeXYZ& svKdTree,
		std::vector<int>& treeLables);

Eigen::Affine3d
optimize(SVMap& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2, gsl_vector* basePose);

Eigen::Vector4f
getNormalizedVectorCode(Eigen::Vector3f vector);

int
getCentroidResultantCode(double norm);

void
createKDTreeForSupervoxels(
		SVMap& supervoxelMap,
		KdTreeXYZ& svKdTree,
		std::vector<int>& labels);

void
transform_get_translation(Eigen::Matrix4d t, double *x, double *y, double *z) {

	*x = t(0,3);
	*y = t(1,3);
	*z = t(2,3);

}

void
transform_get_rotation(Eigen::Matrix4d t, double *x, double *y, double *z) {

	double a = t(2,1);
	double b = t(2,2);
	double c = t(2,0);
	double d = t(1,0);
	double e = t(0,0);

	*x = atan2(a, b);
	*y = asin(-c);
	*z = atan2(d, e);

}

void
printPointClouds(PointCloudT::Ptr scanA, PointCloudT::Ptr transformedScan, string filename) {

	ofstream fout(filename.c_str());

	if (scanA->size() != transformedScan->size()) {
		cerr << "Scan Length not same " << endl;
		return;
	}

	for (int i = 0; i < scanA->size(); ++i) {
		fout << scanA->at(i).x << ' ' << scanA->at(i).y << ' ' << scanA->at(i).z << '\t' << transformedScan->at(i).x << ' ' << transformedScan->at(i).y << ' ' << transformedScan->at(i).z;
		fout << endl;
	}

	fout.close();
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
	Eigen::Affine3d transform = Eigen::Affine3d::Identity();


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

		//		Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
		//				Eigen::Affine3f transform = Eigen::Affine3f::Identity();
		string line;

		for (int i = 0; i < 4; i++) {
			std::getline(in,line);

			std::istringstream sin(line);
			for (int j = 0; j < 4; j++) {
				sin >> transform (i,j);
			}
		}

		cout << "Transformation loaded: " << endl << transform.matrix();
		cout << endl;

		if (!programOptions.showScans && programOptions.test == 0) {

			double x, y, z, roll, pitch, yaw;
			transform_get_translation(transform.matrix(), &x, &y, &z);
			transform_get_rotation(transform.matrix(), &roll, &pitch, &yaw);

			cout << "Calculating base pose" << endl;

			base_pose = gsl_vector_alloc (6);
			gsl_vector_set (base_pose, 0, x);
			gsl_vector_set (base_pose, 1, y);
			gsl_vector_set (base_pose, 2, z);
			gsl_vector_set (base_pose, 3, roll);
			gsl_vector_set (base_pose, 4, pitch);
			gsl_vector_set (base_pose, 5, yaw);

			temp = scan2;

		} else {

			// Input transform should be B rel to A

			transformPointCloud (*scan2, *temp, transform.inverse());
			//			printPointClouds(scan2, temp, "Loaded transform " + transformFile);

			scan2->clear();

		}
	}

	if (programOptions.showScans && programOptions.test == 0) {
		showPointClouds(scan1, temp, "Supervoxel Based MI Viewer: " + transformFile);
		return 0;
	}

	SupervoxelClusteringT super (programOptions.vr, programOptions.sr);
	SVMap supervoxelMapping;
	LabeledLeafMapT labeledLeafMapScan1;

	map <uint32_t, Supervoxel<PointT>::Ptr> supervoxelClusters = initializeVoxels(scan1, temp, super, supervoxelMapping, labeledLeafMapScan1);
	cout << "Number of supervoxels: " << supervoxelMapping.size() << endl;

	Eigen::Affine3d trans_last = Eigen::Affine3d::Identity();
	Eigen::Affine3d trans_new;
	PointCloudT::Ptr transformedScan2;

	KdTreeXYZ svTree;
	std::vector<int> labels;

	SVMap::iterator svItr;

	PointT supervoxelCentroid;
	PointXYZ treePoint;

	createKDTreeForSupervoxels(supervoxelMapping, svTree, labels);

	bool converged = false;
	int iteration = 0;
	double delta;
	bool debug = false;
	double epsilon = 5e-4;
	double epsilon_rot = 2e-3;
	int maxIteration = 10;

	while (!converged) {

		// transform point cloud using trans_last
		if (iteration != 0) {
			transformPointCloud (*temp, *transformedScan2, trans_last);
		} else
			transformedScan2 = temp;

		AdjacencyOctreeT adjTree = super.getOctreeeAdjacency();
		createSuperVoxelMappingForScan2(supervoxelMapping, transformedScan2, labeledLeafMapScan1, adjTree, svTree, labels);

		trans_new = optimize(supervoxelMapping, scan1, transformedScan2, base_pose);

		/* compute the delta from this iteration */
		delta = 0.;
		for(int k = 0; k < 4; k++) {
			for(int l = 0; l < 4; l++) {

				double ratio = 1;
				if(k < 3 && l < 3) {
					// rotation part of the transform
					ratio = 1./epsilon_rot;
				} else {
					ratio = 1./epsilon;
				}

				double diff = trans_last[k][l] - trans_new[k][l];
				double c_delta = ratio*fabs(diff);

				if(c_delta > delta) {
					delta = c_delta;
				}
			}
		}

		if(debug) {
			cout << "delta = " << delta << endl;
		}

		/* check convergence */
		iteration++;
		if(iteration >= maxIteration || delta < 1) {
			converged = true;
		}

		trans_last = trans_new;
		//
	}

}


map <uint32_t, Supervoxel<PointT>::Ptr>
initializeVoxels(
		PointCloudT::Ptr scan1,
		PointCloudT::Ptr scan2,
		SupervoxelClusteringT& super,
		SVMap& supervoxelMapping,
		LabeledLeafMapT& labeledLeafMap) {

	Eigen::Array4f min1, max1, min2, max2;

	PointT minPt, maxPt;

	getMinMax3D(*scan1, minPt, maxPt);

	min1 = minPt.getArray4fMap();
	max1 = maxPt.getArray4fMap();

	getMinMax3D(*scan2, minPt, maxPt);

	min2 = minPt.getArray4fMap();
	max2 = maxPt.getArray4fMap();

	min1 = min1.min(min2);
	max1 = max1.max(max2);

	minPt.x = min1[0]; minPt.y = min1[1]; minPt.z = min1[2];
	maxPt.x = max1[0]; maxPt.y = max1[1]; maxPt.z = max1[2];

	cout << "MinX: " << minPt.x << " MinY: " << minPt.y << " MinZ: " << minPt.z  << endl;
	cout << "MaxX: " << maxPt.x << " MaxY: " << maxPt.y << " MaxZ: " << maxPt.z  << endl;

	super.setVoxelResolution(programOptions.vr);
	super.setSeedResolution(programOptions.sr);
	super.setInputCloud(scan1);
	super.setColorImportance(programOptions.colorWeight);
	super.setSpatialImportance(programOptions.spatialWeight);
	super.setNormalImportance(programOptions.normalWeight);

	super.getOctreeeAdjacency()->customBoundingBox(minPt.x, minPt.y, minPt.z,
			maxPt.x, maxPt.y, maxPt.z);


	// Not being used for now
	map <uint32_t, Supervoxel<PointT>::Ptr> supervoxelClusters;
	super.extract(supervoxelClusters);
	super.getLabeledLeafContainerMap(labeledLeafMap);

	createSuperVoxelMappingForScan1(supervoxelMapping, scan1, labeledLeafMap, super.getOctreeeAdjacency());

	return supervoxelClusters;
}


void
createSuperVoxelMappingForScan1 (SVMap& SVMapping, const typename PointCloudT::Ptr scan, LabeledLeafMapT& labeledLeafMapping, const AdjacencyOctreeT& adjTree) {

	PointCloud<PointT>::iterator scanItr = scan->begin();
	int scanCounter = 0;

	LeafVoxelMapT leafVoxelMap;

	for (;scanItr != scan->end(); ++scanItr, ++scanCounter) {

		PointT a = (*scanItr);

		bool presentInVoxel = adjTree -> isVoxelOccupiedAtPoint(a);

		if (presentInVoxel) {

			typename SupervoxelClusteringT::LeafContainerT* leaf = adjTree -> getLeafContainerAtPoint(a);

			if (leafVoxelMap.find(leaf) != leafVoxelMap.end()) {
				leafVoxelMap[leaf]->getIndexVector()->push_back(scanCounter);
			} else {
				VData::Ptr voxel = boost::shared_ptr<VData>(new VData());
				voxel->getIndexVector()->push_back(scanCounter);
				leafVoxelMap.insert(pair<typename SupervoxelClusteringT::LeafContainerT*, typename VData::Ptr> (leaf, voxel));
			}

		} else {
			// never be the case
			cout << "Not present in voxel"<<endl;
		}
	}

	// leafVoxelMap created for scan1

	LeafVoxelMapT::iterator leafVoxelItr;
	PointCloudT centroidCloud;
	int centroidCloudCounter = 0;

	for (leafVoxelItr = leafVoxelMap.begin(); leafVoxelItr != leafVoxelMap.end(); ++leafVoxelItr, ++centroidCloudCounter) {

		SupervoxelClusteringT::LeafContainerT* leaf = leafVoxelItr->first;
		VData::Ptr voxel = leafVoxelItr->second;

		// compute Centroid
		typename VData::ScanIndexVectorPtr scanIndexVector = voxel->getIndexVector();
		typename VData::ScanIndexVector::iterator indexItr;

		PointT centroid;
		double x(0), y(0), z(0), r(0), g(0), b(0);
		for (indexItr = scanIndexVector->begin(); indexItr != scanIndexVector->end(); ++indexItr) {
			PointT p = scan->at(*indexItr);
			x += p.x;
			y += p.y;
			z += p.z;
			r += p.r;
			g += p.g;
			b += p.b;
		}

		int size = scanIndexVector->size();;
		centroid.x = x / size;
		centroid.y = y / size;
		centroid.z = z / size;
		centroid.r = r / size;
		centroid.g = g / size;
		centroid.g = b / size;

		centroidCloud.push_back(centroid);
		voxel->setCentroid(centroid);
		voxel->setCentroidCloudIndex(centroidCloudCounter);
	}

	cout << "Finding mapping " << endl;
	cout << "Total leaves: " << leafVoxelMap.size() << endl;
	cout << "Total leaves with supervoxels " << labeledLeafMapping.size() << endl;

	int leavesNotFoundWithSupervoxel = 0;
	for (leafVoxelItr = leafVoxelMap.begin(); leafVoxelItr != leafVoxelMap.end(); ++leafVoxelItr) {

		SupervoxelClusteringT::LeafContainerT* leaf = leafVoxelItr->first;
		VData::Ptr voxel = leafVoxelItr->second;
		typename SupervoxelClusteringT::LeafContainerT::const_iterator leafItr;

		// check if leaf exists in the mapping from leaf to label
		if (labeledLeafMapping.find(leaf) != labeledLeafMapping.end()) {

			unsigned int label = labeledLeafMapping[leaf];

			// calculate normal for this leaf
			Eigen::Vector4f params = Eigen::Vector4f::Zero();
			float curvature;
			std::vector<int> indicesToConsider;

			for (leafItr = leaf->cbegin(); leafItr != leaf->cend(); ++leafItr) {
				typename SupervoxelClusteringT::LeafContainerT* neighborLeaf = *leafItr;
				if (leafVoxelMap.find(neighborLeaf) != leafVoxelMap.end() &&
						labeledLeafMapping.find(neighborLeaf) != labeledLeafMapping.end() && labeledLeafMapping[neighborLeaf] == label) {	// same supervoxel
					VData::Ptr neighborVoxel = leafVoxelMap[neighborLeaf];
					indicesToConsider.push_back(neighborVoxel->getCentroidCloudIndex());
				}
			}

			pcl::computePointNormal(centroidCloud, indicesToConsider, params, curvature);

			Eigen::Vector3f normal;
			normal[0] = params[0];
			normal[1] = params[1];
			normal[2] = params[2];

			voxel->setNormal(normal);

			// Normal Calculation end

			// check if SVMapping already contains the supervoxel
			if (SVMapping.find(label) != SVMapping.end()) {
				SVMapping[label]->getVoxelAVector()->push_back(voxel);
			} else {

				SData::Ptr supervoxel = boost::shared_ptr<SData>(new SData());
				supervoxel->setLabel(label);
				supervoxel->getVoxelAVector()->push_back(voxel);

				// Add SV to SVMapping
				SVMapping.insert(pair<uint, typename SData::Ptr>(label, supervoxel));
			}

			// end if

		} else {
			leavesNotFoundWithSupervoxel++;
		}

	}

	cout << "scan 1 leaves without supervoxel: " << leavesNotFoundWithSupervoxel << endl;

	leafVoxelMap.clear();

	cout<<"Finding supervoxel normals " << endl;
	// calculating supervoxel normal
	SVMap::iterator svItr = SVMapping.begin();
	for (; svItr != SVMapping.end(); ++svItr) {

		SData::Ptr supervoxel = svItr->second;
		SData::VoxelVectorPtr voxels = supervoxel->getVoxelAVector();

		Eigen::Vector3f supervoxelNormal = Eigen::Vector3f::Zero();
		PointT supervoxelCentroid;

		double x(0), y(0), z(0), r(0), g(0), b(0);
		typename SData::VoxelVector::iterator voxelItr = voxels->begin();
		for (; voxelItr != voxels->end(); ++voxelItr) {
			supervoxelNormal += (*voxelItr)->getNormal();
			PointT p = (*voxelItr)->getCentroid();
			x += p.x;
			y += p.y;
			z += p.z;
			r += p.r;
			g += p.g;
			b += p.b;
		}

		int voxelSize = voxels->size();
		if (voxelSize != 0) {
			supervoxelCentroid.x = x/voxelSize;
			supervoxelCentroid.y = y/voxelSize;
			supervoxelCentroid.z = z/voxelSize;
			supervoxelCentroid.r = r/voxelSize;
			supervoxelCentroid.g = g/voxelSize;
			supervoxelCentroid.b = b/voxelSize;

			supervoxel->setCentroid(supervoxelCentroid);
		}

		if (!supervoxelNormal.isZero()) {
			supervoxelNormal.normalize();
			supervoxel->setNormal(supervoxelNormal);
		}

	}
}

void
createSuperVoxelMappingForScan2 (
		SVMap& SVMapping,
		const typename PointCloudT::Ptr scan,
		LabeledLeafMapT& labeledLeafMapping,
		const AdjacencyOctreeT& adjTree,
		KdTreeXYZ& svKdTree,
		std::vector<int>& treeLables) {

	SVMap::iterator svItr = SVMapping.begin();
	for (; svItr!=SVMapping.end(); ++svItr) {
		typename SData::Ptr supervoxel = svItr->second;
		supervoxel->clearScanBData();
	}

	PointCloud<PointT>::iterator scanItr;
	int scanCounter = 0;

	LeafVoxelMapT leafVoxelMap;

	for (scanItr = scan->begin();scanItr != scan->end(); ++scanItr, ++scanCounter) {

		PointT a = (*scanItr);

		bool presentInVoxel = adjTree -> isVoxelOccupiedAtPoint(a);

		if (presentInVoxel) {

			typename SupervoxelClusteringT::LeafContainerT* leaf = adjTree -> getLeafContainerAtPoint(a);

			if (leafVoxelMap.find(leaf) != leafVoxelMap.end()) {
				leafVoxelMap[leaf]->getIndexVector()->push_back(scanCounter);
			} else {
				VData::Ptr voxel = boost::shared_ptr<VData>(new VData());
				voxel->getIndexVector()->push_back(scanCounter);
				leafVoxelMap.insert(pair<typename SupervoxelClusteringT::LeafContainerT*, typename VData::Ptr> (leaf, voxel));
			}

		} else
			cout << "Not present in voxel"<<endl;
	}

	// leafVoxelMap created for scan1

	LeafVoxelMapT::iterator leafVoxelItr;
	PointCloudT centroidCloud;
	int centroidCloudCounter = 0;

	for (leafVoxelItr = leafVoxelMap.begin(); leafVoxelItr != leafVoxelMap.end(); ++leafVoxelItr, ++centroidCloudCounter) {

		SupervoxelClusteringT::LeafContainerT* leaf = leafVoxelItr->first;
		VData::Ptr voxel = leafVoxelItr->second;

		// compute Centroid
		typename VData::ScanIndexVectorPtr scanIndexVector = voxel->getIndexVector();
		typename VData::ScanIndexVector::iterator indexItr;

		PointT centroid;
		double x(0), y(0), z(0), r(0), g(0), b(0);
		for (indexItr = scanIndexVector->begin(); indexItr != scanIndexVector->end(); ++indexItr) {
			PointT p = scan->at(*indexItr);
			x += p.x;
			y += p.y;
			z += p.z;
			r += p.r;
			g += p.g;
			b += p.b;
		}

		int size = scanIndexVector->size();
		centroid.x = x / size;
		centroid.y = y / size;
		centroid.z = z / size;
		centroid.r = r / size;
		centroid.g = g / size;
		centroid.g = b / size;

		centroidCloud.push_back(centroid);
		voxel->setCentroid(centroid);
		voxel->setCentroidCloudIndex(centroidCloudCounter);
	}

	// Setup search params
	int NN = SEARCH_SUPERVOXEL_NN;
	PointT voxelCentroid;
	PointXYZ queryPoint;
	std::vector<int> pointIdxNKNSearch(NN);
	std::vector<float> pointNKNSquaredDistance(NN);
	std::vector<int>::iterator intItr;


	for (leafVoxelItr = leafVoxelMap.begin(); leafVoxelItr != leafVoxelMap.end(); ++leafVoxelItr) {

		SupervoxelClusteringT::LeafContainerT* leaf = leafVoxelItr->first;
		VData::Ptr voxel = leafVoxelItr->second;
		typename SupervoxelClusteringT::LeafContainerT::const_iterator leafItr;

		// check if leaf exists in the mapping from leaf to label
		if (labeledLeafMapping.find(leaf) != labeledLeafMapping.end()) {

			unsigned int label = labeledLeafMapping[leaf];

			// check if SVMapping already contains the supervoxel
			if (SVMapping.find(label) != SVMapping.end()) {
				SVMapping[label]->getVoxelAVector()->push_back(voxel);
			} else {

				SData::Ptr supervoxel = boost::shared_ptr<SData>(new SData());
				supervoxel->setLabel(label);
				supervoxel->getVoxelAVector()->push_back(voxel);

				// Add SV to SVMapping
				SVMapping.insert(pair<uint, typename SData::Ptr>(label, supervoxel));
			}

			// end if

		} else {

			// search for the nearest supervoxel with the modified distance function which takes normal into account

			// calculate normal for this leaf
			Eigen::Vector4f params = Eigen::Vector4f::Zero();
			float curvature;
			std::vector<int> indicesToConsider;

			for (leafItr = leaf->cbegin(); leafItr != leaf->cend(); ++leafItr) {
				typename SupervoxelClusteringT::LeafContainerT* neighborLeaf = *leafItr;
				if (leafVoxelMap.find(neighborLeaf) != leafVoxelMap.end()) {
					VData::Ptr neighborVoxel = leafVoxelMap[neighborLeaf];
					indicesToConsider.push_back(neighborVoxel->getCentroidCloudIndex());
				}
			}

			pcl::computePointNormal(centroidCloud, indicesToConsider, params, curvature);

			Eigen::Vector3f normal;
			normal[0] = params[0];
			normal[1] = params[1];
			normal[2] = params[2];

			if (!normal.isZero()) {
				normal.normalize();
				voxel->setNormal(normal);
			}
			// Normal Calculation end

			// search supervoxels for the scan 2 leaf
			voxelCentroid = voxel->getCentroid();

			// Clear prev indices
			pointIdxNKNSearch.clear();
			pointNKNSquaredDistance.clear();

			queryPoint.x = voxelCentroid.x;
			queryPoint.y = voxelCentroid.y;
			queryPoint.z = voxelCentroid.z;

			// search for NN centroids

			if (svKdTree.nearestKSearch (queryPoint,NN, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
			{

				SData::Ptr supervoxel;
				Eigen::Vector3f supervoxelNormal;

				int closestSupervoxelLabel = -1;
				double minDistance = INT_MAX;
				for (intItr = pointIdxNKNSearch.begin(); intItr != pointIdxNKNSearch.end(); ++intItr) {

					int index = *intItr;

					float euclideanD = sqrt(pointNKNSquaredDistance[index]);
					int svLabel = treeLables[index];
					supervoxel = SVMapping[svLabel];
					supervoxelNormal = supervoxel->getNormal();

					double d = acos(normal.dot(supervoxelNormal)) * 2 / M_PI;
					d = 1- log2(1.0 - d);

					d *= euclideanD;
					if (d < minDistance) {
						d = minDistance;
						closestSupervoxelLabel = svLabel;
					}

				}

				if (closestSupervoxelLabel > 0)
					SVMapping[closestSupervoxelLabel]->getVoxelBVector()->push_back(voxel);

			}

			// search for matching normal among the NN supervoxels using new Distance Function:
			// D = (1 - log2( 1 -  acos(SupervoxelNormal dot VoxelNormal) / (Pi/2) )) * Euclidean_Distance
		}
	}

	leafVoxelMap.clear();

}

void
createKDTreeForSupervoxels(
		SVMap& supervoxelMap,
		KdTreeXYZ& svKdTree,
		std::vector<int>& labels) {

	PointCloudXYZ::Ptr svLabelCloud = boost::shared_ptr<PointCloudXYZ>(new PointCloudXYZ());

	SVMap::iterator svItr;

	PointT supervoxelCentroid;
	PointXYZ treePoint;

	for (svItr = supervoxelMap.begin(); svItr != supervoxelMap.end(); ++svItr) {

		SData::Ptr supervoxel = svItr->second;
		int label = svItr->first;

		supervoxelCentroid = supervoxel->getCentroid();

		treePoint.x = supervoxelCentroid.x;
		treePoint.y = supervoxelCentroid.y;
		treePoint.z = supervoxelCentroid.z;
		labels.push_back(label);

		svLabelCloud->push_back(treePoint);
	}

	svKdTree.setInputCloud(svLabelCloud);
}

/*
 *	Function calculates the Mutual Information between ScanA and transformed ScanB
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
 *	Simple MI not working
 *	Attempts
 *
 *		1. Normalizing Mutual Information on the basis of points present in supervoxel.
 */

double
calculateMutualInformation(SVMap& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2) {

	//	bool debug = false;
	//
	//	ofstream debugFile;
	//
	//	if (programOptions.test != 0 && !programOptions.showScans) {
	//		debug = true;
	//		debugFile.open("Normal Info.txt");
	//	}
	//
	//	SVMap::iterator svItr = SVMapping.begin();
	//
	//	map<int, double> normalXProbability;
	//	map<int, double> normalYProbability;
	//	map<string, double> normalXYProbability;
	//
	//	map<int, double> centroidXProbability;
	//	map<int, double> centroidYProbability;
	//	map<string, double> centroidXYProbability;
	//
	//	// Variance Attempt
	//
	//	// Feature 1
	//	map<int, double> variancex_XProbability;
	//	map<int, double> variancex_YProbability;
	//	map<string, double> variancex_XYProbability;
	//
	//	// Feature 2
	//	map<int, double> variancey_XProbability;
	//	map<int, double> variancey_YProbability;
	//	map<string, double> variancey_XYProbability;
	//
	//	// Feature 3
	//	map<int, double> variancez_XProbability;
	//	map<int, double> variancez_YProbability;
	//	map<string, double> variancez_XYProbability;
	//
	//	int size(0); // total overlapping region
	//	double rA(0), rB(0);
	//
	//	unsigned int totalAPointsInOverlappingRegion(0), totalBPointsInOverlappingRegion(0);
	//
	//	for (; svItr!=SVMapping.end(); ++svItr) {
	//
	//		// Write MI Code
	//		int svLabel = svItr->first;
	//		typename SuperVoxelMappingHelper::Ptr supervoxel = svItr->second;
	//
	//		PointNormal supervoxelPointNormalA = supervoxel->getNormalA();
	//		PointNormal supervoxelPointNormalB = supervoxel->getNormalB();
	//
	//		Eigen::Vector3f supervoxelNormalVectorA = supervoxelPointNormalA.getNormalVector3fMap();
	//		Eigen::Vector3f supervoxelNormalVectorB = supervoxelPointNormalB.getNormalVector3fMap();
	//
	//		Eigen::Vector3f supervoxelCentroidVectorA = Eigen::Vector3f::Zero();
	//		Eigen::Vector3f supervoxelCentroidVectorB = Eigen::Vector3f::Zero();
	//
	//		supervoxelCentroidVectorA[0] = supervoxelPointNormalA.x;
	//		supervoxelCentroidVectorA[1] = supervoxelPointNormalA.y;
	//		supervoxelCentroidVectorA[2] = supervoxelPointNormalA.z;
	//
	//		supervoxelCentroidVectorB[0] = supervoxelPointNormalB.x;
	//		supervoxelCentroidVectorB[1] = supervoxelPointNormalB.y;
	//		supervoxelCentroidVectorB[2] = supervoxelPointNormalB.z;
	//
	//		unsigned int counterA = supervoxel->getScanACount();
	//		unsigned int counterB = supervoxel->getScanBCount();
	//
	//		if (counterA > MIN_POINTS_IN_SUPERVOXEL && counterB > MIN_POINTS_IN_SUPERVOXEL) {
	//
	//			// Variance Calculate
	//
	//			double var_x_X(0), var_x_Y(0);
	//			double var_y_X(0), var_y_Y(0);
	//			double var_z_X(0), var_z_Y(0);
	//
	//			bool calculateAVariance = false;
	//			if (supervoxel->getVarianceXCodeA() == 0 ||
	//					supervoxel->getVarianceYCodeA() == 0 ||
	//					supervoxel->getVarianceZCodeA() == 0)
	//				calculateAVariance = true;
	//
	//			SuperVoxelMappingHelper::SimpleVoxelMapPtr voxelMap = supervoxel->getVoxels();
	//			SuperVoxelMappingHelper::SimpleVoxelMap::iterator voxelItr = voxelMap->begin();
	//
	//			for (;voxelItr != voxelMap->end(); ++ voxelItr) {
	//
	//				SimpleVoxelMappingHelper::Ptr voxel = voxelItr->second;
	//				typename SimpleVoxelMappingHelper::ScanIndexVectorPtr indexVectorA = voxel->getScanAIndices();
	//				typename SimpleVoxelMappingHelper::ScanIndexVectorPtr indexVectorB = voxel->getScanBIndices();
	//
	//				typename SimpleVoxelMappingHelper::ScanIndexVector::iterator itr;
	//
	//				if (calculateAVariance) {
	//					for (itr = indexVectorA->begin(); itr != indexVectorA->end(); ++itr) {
	//
	//						double x = scan1->at(*itr).x;
	//						double y = scan1->at(*itr).y;
	//						double z = scan1->at(*itr).z;
	//
	//						var_x_X += square<double> (x-supervoxelPointNormalA.x);
	//						var_y_X += square<double> (y-supervoxelPointNormalA.y);
	//						var_z_X += square<double> (z-supervoxelPointNormalA.z);
	//					}
	//				}
	//
	//				for (itr = indexVectorB->begin(); itr != indexVectorB->end(); ++itr) {
	//
	//					double x = scan2->at(*itr).x;
	//					double y = scan2->at(*itr).y;
	//					double z = scan2->at(*itr).z;
	//
	//					var_x_Y += square<double> (x-supervoxelPointNormalB.x);
	//					var_y_Y += square<double> (y-supervoxelPointNormalB.y);
	//					var_z_Y += square<double> (z-supervoxelPointNormalB.z);
	//				}
	//
	//
	//			}
	//
	//			int varx_XCode = supervoxel->getVarianceXCodeA();
	//			int vary_XCode = supervoxel->getVarianceYCodeA();
	//			int varz_XCode = supervoxel->getVarianceZCodeA();
	//
	//			if (calculateAVariance) {
	//				var_x_X /= counterA;
	//				var_y_X /= counterA;
	//				var_z_X /= counterA;
	//
	//				varx_XCode = getCentroidResultantCode(var_x_X);
	//				vary_XCode = getCentroidResultantCode(var_y_X);
	//				varz_XCode = getCentroidResultantCode(var_z_X);
	//
	//				supervoxel->setVarianceXCodeA(varx_XCode);
	//				supervoxel->setVarianceYCodeA(vary_XCode);
	//				supervoxel->setVarianceZCodeA(varz_XCode);
	//			}
	//
	//			var_x_Y /= counterB;
	//			var_y_Y /= counterB;
	//			var_z_Y /= counterB;
	//
	//			int varx_YCode = getCentroidResultantCode(var_x_Y);
	//			int vary_YCode = getCentroidResultantCode(var_y_Y);
	//			int varz_YCode = getCentroidResultantCode(var_z_Y);
	//
	//			supervoxel->setVarianceXCodeB(varx_YCode);
	//			supervoxel->setVarianceYCodeB(vary_YCode);
	//			supervoxel->setVarianceZCodeB(varz_YCode);
	//
	//			string varx_XYCode = boost::str(boost::format("%d_%d")%varx_XCode%varx_YCode);
	//			string vary_XYCode = boost::str(boost::format("%d_%d")%vary_XCode%vary_YCode);
	//			string varz_XYCode = boost::str(boost::format("%d_%d")%varz_XCode%varz_YCode);
	//
	//			supervoxel->setVarianceXCodeAB(varx_XYCode);
	//			supervoxel->setVarianceYCodeAB(vary_XYCode);
	//			supervoxel->setVarianceZCodeAB(varz_XYCode);
	//
	//			// Variance X Features
	//			if (variancex_XProbability.find(varx_XCode) != variancex_XProbability.end()) {
	//				variancex_XProbability[varx_XCode] += 1;
	//			}  else {
	//				variancex_XProbability.insert(pair<int, double> (varx_XCode, 1.0));
	//			}
	//
	//			if (variancey_XProbability.find(vary_XCode) != variancey_XProbability.end()) {
	//				variancey_XProbability[vary_XCode] += 1;
	//			}  else {
	//				variancey_XProbability.insert(pair<int, double> (vary_XCode, 1.0));
	//			}
	//
	//			if (variancez_XProbability.find(varz_XCode) != variancez_XProbability.end()) {
	//				variancez_XProbability[varz_XCode] += 1;
	//			}  else {
	//				variancez_XProbability.insert(pair<int, double> (varz_XCode, 1.0));
	//			}
	//
	//			// Variance Y Features
	//			if (variancex_YProbability.find(varx_YCode) != variancex_YProbability.end()) {
	//				variancex_YProbability[varx_YCode] += 1;
	//			}  else {
	//				variancex_YProbability.insert(pair<int, double> (varx_YCode, 1.0));
	//			}
	//
	//			if (variancey_YProbability.find(vary_YCode) != variancey_YProbability.end()) {
	//				variancey_YProbability[vary_YCode] += 1;
	//			}  else {
	//				variancey_YProbability.insert(pair<int, double> (vary_YCode, 1.0));
	//			}
	//
	//			if (variancez_YProbability.find(varz_YCode) != variancez_YProbability.end()) {
	//				variancez_YProbability[varz_YCode] += 1;
	//			}  else {
	//				variancez_YProbability.insert(pair<int, double> (varz_YCode, 1.0));
	//			}
	//
	//			// Variance XY Features
	//			if (variancex_XYProbability.find(varx_XYCode) != variancex_XYProbability.end()) {
	//				variancex_XYProbability[varx_XYCode] += 1;
	//			}  else {
	//				variancex_XYProbability.insert(pair<string, double> (varx_XYCode, 1.0));
	//			}
	//
	//			if (variancey_XYProbability.find(vary_XYCode) != variancey_XYProbability.end()) {
	//				variancey_XYProbability[vary_XYCode] += 1;
	//			}  else {
	//				variancey_XYProbability.insert(pair<string, double> (vary_XYCode, 1.0));
	//			}
	//
	//			if (variancez_XYProbability.find(varz_XYCode) != variancez_XYProbability.end()) {
	//				variancez_XYProbability[varz_XYCode] += 1;
	//			}  else {
	//				variancez_XYProbability.insert(pair<string, double> (varz_XYCode, 1.0));
	//			}
	//
	//			// End Variance computation
	//
	//			totalAPointsInOverlappingRegion += counterA;
	//			totalBPointsInOverlappingRegion += counterB;
	//			size++;
	//
	//			//			if (!supervoxelNormalVectorA.isZero()) {
	//			//				supervoxelNormalVectorA.normalize();
	//			//			}
	//			//
	//			//			if (!supervoxelNormalVectorB.isZero()) {
	//			//				supervoxelNormalVectorB.normalize();
	//			//			}
	//			//
	//			//			rA = supervoxelCentroidVectorA.norm();
	//			//			rB = supervoxelCentroidVectorB.norm();
	//			//
	//			//			int normalCodeA(0), normalCodeB(0), centroidCodeA(0), centroidCodeB(0);
	//			//			Eigen::Vector4f normalCodeVectorA, normalCodeVectorB; // centroidCodeVectorA, centroidCodeVectorB;
	//			//
	//			//			normalCodeA = supervoxel->getNormalCodeA();
	//			//			centroidCodeA = supervoxel->getCentroidCodeA();
	//			//
	//			//			// cache A code
	//			//			if (normalCodeA == 0) {
	//			//				normalCodeVectorA = getNormalizedVectorCode(supervoxelNormalVectorA);
	//			//				normalCodeA = normalCodeVectorA[3];
	//			//				supervoxel->setNormalCodeA(normalCodeA);
	//			//			}
	//			//
	//			//			if (centroidCodeA == 0) {
	//			//				//				centroidCodeVectorA = getNormalizedVectorCode(supervoxelCentroidVectorA);
	//			//				//				centroidCodeA = centroidCodeVectorA[3];
	//			//				centroidCodeA = getCentroidResultantCode(rA);
	//			//				supervoxel->setCentroidCodeA(centroidCodeA);
	//			//			}
	//			//
	//			//			normalCodeVectorB = getNormalizedVectorCode(supervoxelNormalVectorB);
	//			//			normalCodeB = normalCodeVectorB[3];
	//			//			supervoxel->setNormalCodeB(normalCodeB);
	//			//
	//			//			centroidCodeB = getCentroidResultantCode(rB);
	//			//			supervoxel->setCentroidCodeB(centroidCodeB);
	//			//
	//			//			if (normalXProbability.find(normalCodeA) != normalXProbability.end()) {
	//			//				normalXProbability[normalCodeA] += 1;
	//			//			}  else {
	//			//				normalXProbability.insert(pair<int, double> (normalCodeA, 1.0));
	//			//			}
	//			//
	//			//			if (centroidXProbability.find(centroidCodeA) != centroidXProbability.end()) {
	//			//				centroidXProbability[centroidCodeA] += 1;
	//			//			} else {
	//			//				centroidXProbability.insert(pair<int, double> (centroidCodeA, 1.0));
	//			//			}
	//			//
	//			//			if (normalYProbability.find(normalCodeB) != normalYProbability.end())
	//			//				normalYProbability[normalCodeB]+= 1;
	//			//			else
	//			//				normalYProbability.insert(pair<int, double> (normalCodeB, 1.0));
	//			//
	//			//			if (centroidYProbability.find(centroidCodeB) != centroidYProbability.end()) {
	//			//				centroidYProbability[centroidCodeB] += 1;
	//			//			} else {
	//			//				centroidYProbability.insert(pair<int, double> (centroidCodeB, 1.0));
	//			//			}
	//			//
	//			//			string centroidCodePair = boost::str(boost::format("%d_%d")%centroidCodeA%centroidCodeB);
	//			//			string normalCodePair = boost::str(boost::format("%d_%d")%normalCodeA%normalCodeB);
	//			//
	//			//			supervoxel->setNormalCodeAB(normalCodePair);
	//			//			supervoxel->setCentroidCodeAB(centroidCodePair);
	//			//
	//			//			if (normalXYProbability.find(normalCodePair) != normalXYProbability.end())
	//			//				normalXYProbability[normalCodePair] += 1;
	//			//			else
	//			//				normalXYProbability.insert(pair<string, double> (normalCodePair, 1.0));
	//			//
	//			//			if (centroidXYProbability.find(centroidCodePair) != centroidXYProbability.end())
	//			//				centroidXYProbability[centroidCodePair] += 1;
	//			//			else
	//			//				centroidXYProbability.insert(pair<string, double> (centroidCodePair, 1.0));
	//			//
	//			//			if (debug) {
	//			//
	//			//				debugFile << svLabel << endl;
	//			//
	//			//				debugFile << "Normals" << endl;
	//			//
	//			//				debugFile << "A code: " << endl;
	//			//				debugFile << supervoxelNormalVectorA << endl;
	//			//				debugFile << normalCodeVectorA << endl;
	//			//
	//			//				debugFile << "B code: " << endl;
	//			//				debugFile << supervoxelNormalVectorB << endl;
	//			//				debugFile << normalCodeVectorB << endl;
	//			//
	//			//				debugFile << "Centroids" << endl;
	//			//
	//			//				debugFile << "A code: " << endl;
	//			//				debugFile << rA << '\t' << centroidCodeA << endl;
	//			//
	//			//				debugFile << "B code: " << endl;
	//			//				debugFile << rB << '\t' << centroidCodeB << endl;
	//			//
	//			//			}
	//
	//			//			// Normal Angle Info
	//			//			double theta;
	//			//			double dotPro = svNormA.dot(svNormB);
	//			//			theta = (180.00 / M_PI) * acos(dotPro);
	//			//
	//			//			float normX = svNormA[0];
	//			//			float normY = svNormA[1];
	//			//			float normZ = svNormA[2];
	//			//			cout<<boost::format("%d A: %d %f %f %f")%svItr->first%counterA%normX%normY%normZ<<endl;
	//			//
	//			//			normX = svNormB[0];
	//			//			normY = svNormB[1];
	//			//			normZ = svNormB[2];
	//			//
	//			//			cout<<boost::format("%d B: %d %f %f %f")%svItr->first%counterB%normX%normY%normZ<<endl;
	//			//
	//			//			cout << svItr->first <<" Theta: "<< theta << endl;
	//		}
	//
	//	}
	//
	//	// Calculating probabilities for all norm codes
	//	map<int, double>::iterator itr;
	//
	//	// Calculating prob for all events of X for feeatures x,y,z
	//	for (itr = variancex_XProbability.begin(); itr != variancex_XProbability.end(); ++itr) {
	//		double x = ((double)itr->second) / size;
	//		itr->second = x;
	//	}
	//
	//	for (itr = variancey_XProbability.begin(); itr != variancey_XProbability.end(); ++itr) {
	//		double x = ((double)itr->second) / size;
	//		itr->second = x;
	//	}
	//
	//	for (itr = variancez_XProbability.begin(); itr != variancez_XProbability.end(); ++itr) {
	//		double x = ((double)itr->second) / size;
	//		itr->second = x;
	//	}
	//
	//	// Calculating prob for all events of Y for feeatures x,y,z
	//	for (itr = variancex_YProbability.begin(); itr != variancex_YProbability.end(); ++itr) {
	//		double x = ((double)itr->second) / size;
	//		itr->second = x;
	//	}
	//
	//	for (itr = variancey_YProbability.begin(); itr != variancey_YProbability.end(); ++itr) {
	//		double x = ((double)itr->second) / size;
	//		itr->second = x;
	//	}
	//
	//	for (itr = variancez_YProbability.begin(); itr != variancez_YProbability.end(); ++itr) {
	//		double x = ((double)itr->second) / size;
	//		itr->second = x;
	//	}
	//
	//
	//	//	for (itr = normalXProbability.begin(); itr != normalXProbability.end(); ++itr) {
	//	//		double x = ((double)itr->second) / size;
	//	//		itr->second = x;
	//	//	}
	//	//
	//	//	for (itr = centroidXProbability.begin(); itr != centroidXProbability.end(); ++itr) {
	//	//		double x = ((double)itr->second) / size;
	//	//		itr->second = x;
	//	//	}
	//	//
	//	//	for (itr = normalYProbability.begin(); itr != normalYProbability.end(); ++itr) {
	//	//		double y = ((double)itr->second) / size;
	//	//		itr->second = y;
	//	//	}
	//	//
	//	//	for (itr = centroidYProbability.begin(); itr != centroidYProbability.end(); ++itr) {
	//	//		double y = ((double)itr->second) / size;
	//	//		itr->second = y;
	//	//	}
	//
	//	map<string, double>::iterator xyItr;
	//
	//	// Calculating prob for all events of XY for features x,y,z
	//
	//	for (xyItr = variancex_XYProbability.begin(); xyItr != variancex_XYProbability.end(); ++xyItr) {
	//		double xy = ((double)xyItr->second) / size;
	//		xyItr->second = xy;
	//	}
	//
	//	for (xyItr = variancey_XYProbability.begin(); xyItr != variancey_XYProbability.end(); ++xyItr) {
	//		double xy = ((double)xyItr->second) / size;
	//		xyItr->second = xy;
	//	}
	//
	//	for (xyItr = variancez_XYProbability.begin(); xyItr != variancez_XYProbability.end(); ++xyItr) {
	//		double xy = ((double)xyItr->second) / size;
	//		xyItr->second = xy;
	//	}
	//
	//	//	for (xyItr = normalXYProbability.begin(); xyItr != normalXYProbability.end(); ++xyItr) {
	//	//		double xy = ((double)xyItr->second) / size;
	//	//		xyItr->second = xy;
	//	//	}
	//	//
	//	//	for (xyItr = centroidXYProbability.begin(); xyItr != centroidXYProbability.end(); ++xyItr) {
	//	//		double xy = ((double)xyItr->second) / size;
	//	//		xyItr->second = xy;
	//	//	}
	//
	//
	//	// calculate MI for overlapping supervoxels using normalXProbability, randomY and normalXYProbability
	//
	//	double hX(0), hY(0), hXY(0);
	//
	//	svItr = SVMapping.begin();
	//	for (; svItr != SVMapping.end(); ++svItr) {
	//
	//		SuperVoxelMappingHelper::Ptr supervoxel = svItr->second;
	//
	//		unsigned int counterA = supervoxel->getScanACount();
	//		unsigned int counterB = supervoxel->getScanBCount();
	//
	//		if (counterA > MIN_POINTS_IN_SUPERVOXEL && counterB > MIN_POINTS_IN_SUPERVOXEL) {
	//
	//			// MI calculation using varX, varY, varZ as features
	//
	//			int varxACode = supervoxel->getVarianceXCodeA();
	//			int varxBCode = supervoxel->getVarianceXCodeB();
	//			string varxABCode = supervoxel->getVarianceXCodeAB();
	//
	//			int varyACode = supervoxel->getVarianceYCodeA();
	//			int varyBCode = supervoxel->getVarianceYCodeB();
	//			string varyABCode = supervoxel->getVarianceYCodeAB();
	//
	//
	//			int varzACode = supervoxel->getVarianceZCodeA();
	//			int varzBCode = supervoxel->getVarianceZCodeB();
	//			string varzABCode = supervoxel->getVarianceZCodeAB();
	//
	//			double varxAPro = variancex_XProbability.at(varxACode);
	//			double varyAPro = variancey_XProbability.at(varyACode);
	//			double varzAPro = variancez_XProbability.at(varzACode);
	//
	//			double varxBPro = variancex_YProbability.at(varxBCode);
	//			double varyBPro = variancey_YProbability.at(varyBCode);
	//			double varzBPro = variancez_YProbability.at(varzBCode);
	//
	//			double varxABPro = variancex_XYProbability.at(varxABCode);
	//			double varyABPro = variancey_XYProbability.at(varyABCode);
	//			double varzABPro = variancez_XYProbability.at(varzABCode);
	//
	//			hX += varxAPro * log(varxAPro) + varyAPro * log(varyAPro) + varzAPro * log(varzAPro);
	//			hY += varxBPro * log(varxBPro) + varyBPro * log(varyBPro) + varzBPro * log(varzBPro);
	//			hXY += varxABPro * log(varxABPro) + varyABPro * log(varyABPro) + varzABPro * log(varzABPro);
	//
	//			// Commenting out MI calculation using Normal and Centroid as Features
	//
	//			//			int normalCodeA = supervoxel->getNormalCodeA();
	//			//			int normalCodeB = supervoxel->getNormalCodeB();
	//			//			string normalCodeAB = supervoxel->getNormalCodeAB();
	//			//
	//			//			int centroidCodeA = supervoxel->getCentroidCodeA();
	//			//			int centroidCodeB = supervoxel->getCentroidCodeB();
	//			//			string centroidCodeAB = supervoxel->getCentroidCodeAB();
	//			//
	//			//			double normalProX = normalXProbability.at(normalCodeA);
	//			//			double normalProY = normalYProbability.at(normalCodeB);
	//			//			double normalProXY = normalXYProbability.at(normalCodeAB);
	//			//
	//			//			double centroidProX = centroidXProbability.at(centroidCodeA);
	//			//			double centroidProY = centroidYProbability.at(centroidCodeB);
	//			//			double centroidProXY = centroidXYProbability.at(centroidCodeAB);
	//			//
	//			//			//			hX += /*normalProX * log(normalProX) +*/ centroidProX * log(centroidProX);
	//			//			//			hY += /*normalProY * log(normalProY) +*/ centroidProY * log(centroidProY);
	//			//			//			hXY += /*normalProXY * log(normalProXY) +*/ centroidProXY * log(centroidProXY);
	//			//
	//			//
	//			//			hY += normalProY * log(normalProY) + centroidProY * log(centroidProY);
	//			//			hXY += normalProXY * log(normalProXY) + centroidProXY * log(centroidProXY);
	//
	//		}
	//
	//	}
	//
	//	hX *= -1;
	//	hY *= -1;
	//	hXY *= -1;
	//
	double mi = 0;//hX + hY - hXY;
	//	double nmi = (hX + hY) / hXY;
	//
	//	cout << "H(X) = " << hX << '\t' << "H(Y) = " << hY << '\t' << "H(X,Y) = " << hXY << '\t' << "MI(X,Y) = " << mi << '\t' << "NMI(X,Y) = " << nmi << endl;
	//
	//	if (debug)
	//		debugFile.close();

	return mi;
}

struct MI_Opti_Data{
	SVMap* svMap;
	PointCloudT::Ptr scan1;
	PointCloudT::Ptr scan2;
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
 *	CommonSupervoxels will be used for MI calculation
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
	//	LabeledLeafMapT* labeledLeafMap = miOptiData->labeledLeafMap;
	//	AdjacencyOctreeT* adjTree = miOptiData->adjTree;

	// Create Transformation
	Eigen::Affine3d transform = Eigen::Affine3d::Identity();
	transform.translation() << x,y,z;
	transform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
	transform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
	transform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));

	//	cout << "Applying Transform " << endl << transform.matrix() << endl;

	// Transform point cloud
	pcl::transformPointCloud(*scan2, *transformedScan2, transform);

	// Clear SVMap for new scan2 properties
	SVMap::iterator svItr = SVMapping->begin();

	for (; svItr != SVMapping->end(); ++svItr) {
		int label = svItr->first;
		//		SuperVoxelMappingHelper::Ptr svMapHelper = svItr->second;
		//		svMapHelper->clearScanBData();
	}

	// recreate map for scan2
	//	createSuperVoxelMappingForScan2(*SVMapping, transformedScan2, *labeledLeafMap, *adjTree);

	// compute Voxel Data for scan 2
	//	computeVoxelCentroidScan2(*SVMapping, transformedScan2, *labeledLeafMap);


	double mi = calculateMutualInformation(*SVMapping, scan1, transformedScan2);

	cout << "MI Function Called with refreshed values" << mi << endl;


	return -mi;
}

Eigen::Affine3d optimize(SVMap& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2, gsl_vector* baseX) {

	MI_Opti_Data* mod = new MI_Opti_Data();
	mod->scan1 = scan1;
	mod->scan2 = scan2;
	mod->svMap = &SVMapping;

	const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;

	//	const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_vector_bfgs2;
	gsl_multimin_fminimizer *s = NULL;

	gsl_vector *ss;
	gsl_multimin_function minex_func;

	size_t iter = 0;
	int status;
	double size;

	/* Set  initial step sizes to 1 */
	ss = gsl_vector_alloc (6);
	gsl_vector_set (ss, 0, 0.5);
	gsl_vector_set (ss, 1, 0.5);
	gsl_vector_set (ss, 2, 0.5);
	gsl_vector_set (ss, 3, 0.1);
	gsl_vector_set (ss, 4, 0.1);
	gsl_vector_set (ss, 5, 0.1);

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

		printf("%5d f() = %7.3f size = %d\n",
				iter,
				s->fval,
				(int)size);

		if (status == GSL_SUCCESS) {

			cout << "FLast = " << mi_f(s->x, mod) << endl;
			//			printSVMapDetails(SVMapping, "trans_mi");

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

			Eigen::Affine3d resultantTransform = Eigen::Affine3d::Identity();
			resultantTransform.translation() << tx, ty, tz;
			resultantTransform.rotate (Eigen::AngleAxisd (roll, Eigen::Vector3d::UnitX()));
			resultantTransform.rotate (Eigen::AngleAxisd (pitch, Eigen::Vector3d::UnitY()));
			resultantTransform.rotate(Eigen::AngleAxisd (yaw, Eigen::Vector3d::UnitZ()));

			cout << "Resulting Transformation: " << endl << resultantTransform.inverse().matrix();
			cout << endl;

			return resultantTransform.inverse();
		}

	} while(status == GSL_CONTINUE && iter < 100);

	//	gsl_vector_free(baseX);
	gsl_vector_free(ss);
	gsl_multimin_fminimizer_free(s);

	return Eigen::Affine3d::Identity();
}

/*
 * 	Returns a unique Normal Vector code for a group of normalized vectors in the range
 * 	[x,y,z] - [x+dx, y+dy, z+dy]
 *
 * 	NormalVectorCode is a label
 * 	Its value is not of any significance
 *
 *	-1.0 -> 0
 *	-0.8 -> 1
 *	-0.6 -> 2
 *	and so on
 *
 */

Eigen::Vector4f
getNormalizedVectorCode(Eigen::Vector3f vector) {

	Eigen::Vector4f resultantCode = Eigen::Vector4f::Zero();

	float x = vector[0];
	float y = vector[1];
	float z = vector[2];

	if (abs(x) > 1 || abs(y) > 1 || abs(z) > 1)
		return resultantCode;

	int a(0), b(0), c(0);
	float dx(NORM_DX), dy(NORM_DY), dz(NORM_DZ), diff;

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

	//	diff = dx - abs(x);
	//	if (2 * diff > NORM_DX) // Moving to a closer step
	//		a++;


	while (dy < abs(y)) {
		dy += NORM_DY;
		b++;
	}

	//	diff = dy - abs(y);
	//	if (2*diff > NORM_DY)
	//		b++;

	while (dz < abs(z)) {
		dz += NORM_DZ;
		c++;
	}

	//	diff = dz - abs(z);
	//	if (2*diff > NORM_DZ)
	//		c++;

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

	resultantCode[0] = a;
	resultantCode[1] = b;
	resultantCode[2] = c;
	resultantCode[3] = code;

	return resultantCode;
}

/*
 *
 */
int
getCentroidResultantCode(double norm) {

	int a(0);
	double dR = NORM_R;
	double dNorm = dR;

	while (dNorm < norm) {
		a++;
		dNorm += dR;
	}

	double diff = dNorm - norm;
	diff *= 2;

	if (diff - dR < 0)
		a++;

	return a;
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

//void
//showTestSuperVoxel(SVMap& SVMapping, PointCloudT::Ptr scan1, PointCloudT::Ptr scan2) {
//
//	// Display all supervoxels with count A and count B
//
//	SVMap::iterator svItr = SVMapping.begin();
//
//	for (; svItr!=SVMapping.end(); ++svItr) {
//
//		// Write MI Code
//		int svLabel = svItr->first;
//		typename SuperVoxelMappingHelper::Ptr supervoxel = svItr->second;
//
//		SuperVoxelMappingHelper::SimpleVoxelMapPtr voxelMap = supervoxel->getVoxels();
//		SuperVoxelMappingHelper::SimpleVoxelMap::iterator voxelItr = voxelMap->begin();
//
//		int counterA(0), counterB(0);
//		for (; voxelItr != voxelMap -> end(); ++ voxelItr) {
//
//			SimpleVoxelMappingHelper::Ptr voxel = (*voxelItr).second;
//			counterA += voxel->getScanAIndices()->size();
//			counterB += voxel->getScanBIndices()->size();
//		}
//
//		cout << svLabel << '\t' << "A: " << counterA << '\t' << "B: " << counterB << endl;
//
//	}
//
//	//
//	int SV = programOptions.test;
//
//	typename PointCloudT::Ptr newCloud (new PointCloudT);
//
//	SuperVoxelMappingHelper::SimpleVoxelMap::iterator vxlItr = SVMapping[SV] -> getVoxels() -> begin();
//	int scanACounter(0), scanBCounter(0);
//
//	for (; vxlItr != SVMapping[SV] -> getVoxels() ->end(); ++vxlItr) {
//
//		SimpleVoxelMappingHelper::Ptr voxel = (*vxlItr).second;
//
//		typename SimpleVoxelMappingHelper::ScanIndexVectorPtr scanAVector = voxel->getScanAIndices();
//		typename SimpleVoxelMappingHelper::ScanIndexVectorPtr scanBVector = voxel->getScanBIndices();
//
//		typename SimpleVoxelMappingHelper::ScanIndexVector::iterator vi = scanAVector -> begin();
//		for (; vi != scanAVector -> end(); ++vi, ++scanACounter) {
//
//			PointT p = scan1->at(*vi);
//
//			p.r = 255;
//			p.g = 0;
//			p.b = 0;
//
//			newCloud->push_back(p);
//		}
//
//		vi = scanBVector -> begin();
//
//		for (; vi != scanBVector -> end(); ++vi, ++scanBCounter) {
//
//			PointT p = scan2->at(*vi);
//
//			p.r = 0;
//			p.g = 255;
//			p.b = 0;
//
//			newCloud->push_back(p);
//		}
//
//
//
//	}
//
//	showPointCloud(newCloud);
//}

