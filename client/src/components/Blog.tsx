import React from 'react';
import { Box, Heading, Text } from '@chakra-ui/react';
import BlogType from '../types/BlogType';

const Blog = (blog : BlogType ) => {
    return (
        <Box p={4} shadow="md" borderWidth="1px" rounded="md"  color="gray.700">
            <Heading as="h2" size="md" mb={2}>
                {blog.title}
            </Heading>
            <Text mb={4} fontSize="sm">
                {blog.description}
            </Text>
        </Box>
    );
};

export default Blog;
