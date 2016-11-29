/*
 * supervoxel_mapping.hpp
 *
 *  Created on: 24-Nov-2016
 *      Author: root
 */

#ifndef SUPERVOXEL_MAPPING_HPP_
#define SUPERVOXEL_MAPPING_HPP_

#include <vector>
#include <boost/shared_ptr.hpp>
#include <pcl/point_types.h>
#include "supervoxel_cluster_search.h"
#include <pcl/octree/octree_key.h>

struct MOctreeKey {

	MOctreeKey(pcl::octree::OctreeKey octreeKey) {
		this->key = octreeKey;
	}

	bool operator<(const MOctreeKey& arg) const
	{
		if (this->key.x < arg.key.x)
			return true;
		else if (this->key.x == arg.key.x && this->key.y < arg.key.y)
			return true;
		else if (this->key.x == arg.key.x && this->key.y == arg.key.y && this->key.z < arg.key.z)
			return true;

		return false;
	}

	typename pcl::octree::OctreeKey key;
};

class SimpleVoxelMappingHelper {

public:

	typedef boost::shared_ptr<SimpleVoxelMappingHelper> Ptr;
	typedef std::vector<int> ScanIndexVector;
	typedef boost::shared_ptr<std::vector<int> > ScanIndexVectorPtr;

	SimpleVoxelMappingHelper();

	~SimpleVoxelMappingHelper();

	ScanIndexVectorPtr
	getScanAIndices();

	ScanIndexVectorPtr
	getScanBIndices();

	void
	setNormalA(typename pcl::PointNormal normal) {
		this->normalA = normal;
	}

	void
	setNormalB(typename pcl::PointNormal normal) {
		this->normalB = normal;
	}

	typename pcl::PointNormal
	getNormalA() {
		return this->normalA;
	}

	typename pcl::PointNormal
	getNormalB() {
		return this->normalB;
	}

//	typename pcl::RGB
//	getrgbA() {
//		return rgbA;
//	}
//
//	typename pcl::RGB
//	getrgbB() {
//		return rgbB;
//	}
//
//	void
//	setrgbA(typename pcl::RGB rgb) {
//		this->rgbA = rgb;
//	}
//
//	void
//	setrgbB(typename pcl::RGB rgb) {
//		this->rgbB = rgb;
//	}

//	void
//	setLeaf(typename pcl::SupervoxelClustering<pcl::PointXYZRGBA>::LeafContainerT* leaf) {
//		this->leafPtr = leaf;
//	}
//
//	typename pcl::SupervoxelClustering<pcl::PointXYZRGBA>::LeafContainerT*
//	getLeaf() {
//		return leafPtr;
//	}

	pcl::PointXYZRGBA
	getCentroidA() {
		return centroidA;
	}

	pcl::PointXYZRGBA
	getCentroidB() {
		return centroidB;
	}

	void
	setCentroidA(pcl::PointXYZRGBA centroid) {
		this->centroidA = centroid;
	}

	void
	setCentroidB(pcl::PointXYZRGBA centroid) {
		this->centroidB = centroid;
	}

	void
	setIndex(int index) {
		idx = index;
	}

	int
	getIndex() {
		return idx;
	}

private:

	ScanIndexVectorPtr indicesAPtr;
	ScanIndexVectorPtr indicesBPtr;
	typename pcl::PointNormal normalA;
	typename pcl::PointNormal normalB;
//	typename pcl::RGB rgbA;
//	typename pcl::RGB rgbB;
	pcl::PointXYZRGBA centroidA;
	pcl::PointXYZRGBA centroidB;
//	typename pcl::SupervoxelClustering<pcl::PointXYZRGBA>::LeafContainerT* leafPtr;
	int idx;

};

class SuperVoxelMappingHelper {

public:

	typedef typename pcl::octree::OctreeKey OctreeKeyT;
	//	typedef typename std::map<MOctreeKey, typename SimpleVoxelMappingHelper::Ptr> SimpleVoxelMap;
	typedef typename std::map<pcl::SupervoxelClustering<pcl::PointXYZRGBA>::LeafContainerT*, typename SimpleVoxelMappingHelper::Ptr> SimpleVoxelMap;
	typedef typename boost::shared_ptr<SimpleVoxelMap> SimpleVoxelMapPtr;
	typedef typename boost::shared_ptr<SuperVoxelMappingHelper> Ptr;

	SuperVoxelMappingHelper(unsigned int);

	~SuperVoxelMappingHelper();

	unsigned int
	getSuperVoxelLabel();

	void
	setNormalA(typename pcl::PointNormal normal) {
		this->centroidNormalA = normal;
	}

	void
	setNormalB(typename pcl::PointNormal normal) {
		this->centroidNormalB = normal;
	}

	typename pcl::PointNormal
	getNormalA() {
		return this->centroidNormalA;
	}

	typename pcl::PointNormal
	getNormalB() {
		return this->centroidNormalB;
	}

	typename pcl::RGB
	getrgbA() {
		return centroidRGBA;
	}

	typename pcl::RGB
	getrgbB() {
		return centroidRGBB;
	}

	void
	setrgbA(typename pcl::RGB rgb) {
		this->centroidRGBA = rgb;
	}

	void
	setrgbB(typename pcl::RGB rgb) {
		this->centroidRGBB = rgb;
	}

	SimpleVoxelMapPtr
	getVoxels();

private :

	unsigned int label;
	typename pcl::PointNormal centroidNormalA;
	typename pcl::PointNormal centroidNormalB;
	typename pcl::RGB centroidRGBA;
	typename pcl::RGB centroidRGBB;
	SimpleVoxelMapPtr voxelMap;
};

#endif /* SUPERVOXEL_MAPPING_HPP_ */
