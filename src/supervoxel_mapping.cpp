#include "supervoxel_mapping.hpp"

SuperVoxelMappingHelper::SuperVoxelMappingHelper(unsigned int label) {
	this->label = label;
	SimpleVoxelMapPtr p;
	voxelMap.reset(new typename SuperVoxelMappingHelper::SimpleVoxelMap());
	//	voxelMap->
}

SuperVoxelMappingHelper::~SuperVoxelMappingHelper() {

}

typename SuperVoxelMappingHelper::SimpleVoxelMapPtr
SuperVoxelMappingHelper::getVoxels() {
	return voxelMap;
}

SimpleVoxelMappingHelper::~SimpleVoxelMappingHelper() {}

SimpleVoxelMappingHelper::SimpleVoxelMappingHelper() {
	indicesAPtr.reset(new typename SimpleVoxelMappingHelper::ScanIndexVector());
	indicesBPtr.reset(new typename SimpleVoxelMappingHelper::ScanIndexVector());
	idxA = 0;
	idxB = 0;
};

typename SimpleVoxelMappingHelper::ScanIndexVectorPtr
SimpleVoxelMappingHelper::getScanBIndices() {
	return indicesBPtr;
}

typename SimpleVoxelMappingHelper::ScanIndexVectorPtr
SimpleVoxelMappingHelper::getScanAIndices() {
	return indicesAPtr;
}
