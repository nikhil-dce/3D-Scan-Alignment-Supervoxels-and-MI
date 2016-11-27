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

class SuperVoxelMapping {

public:

	typedef boost::shared_ptr<SuperVoxelMapping> Ptr;

	SuperVoxelMapping(unsigned int);

	~SuperVoxelMapping();

	unsigned int
	getSuperVoxelLabel();

	typename boost::shared_ptr<std::vector<unsigned int> >
	getScanAIndices();

	typename boost::shared_ptr<std::vector<unsigned int> >
	getScanBIndices();

private :

	unsigned int label;
	typename boost::shared_ptr<std::vector<unsigned int> > indicesAPtr;
	typename boost::shared_ptr<std::vector<unsigned int> > indicesBPtr;

};



#endif /* SUPERVOXEL_MAPPING_HPP_ */
