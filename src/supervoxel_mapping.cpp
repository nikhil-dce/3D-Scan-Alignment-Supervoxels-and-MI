#include "supervoxel_mapping.hpp"

SuperVoxelMapping::SuperVoxelMapping(unsigned int label) {
	this->label = label;
	indicesAPtr.reset(new std::vector<unsigned int>);
	indicesBPtr.reset(new std::vector<unsigned int>);
}

SuperVoxelMapping::~SuperVoxelMapping() {

}

typename boost::shared_ptr<std::vector<unsigned int> >
SuperVoxelMapping::getScanAIndices() {
	return indicesAPtr;
}

typename boost::shared_ptr<std::vector<unsigned int> >
SuperVoxelMapping::getScanBIndices() {
	return indicesBPtr;
}
