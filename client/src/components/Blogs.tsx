import React, { useState, useEffect } from 'react';
import {send_request} from "../scripts/request.ts";
import Blog from "./Blog";
import {Box, Heading, SimpleGrid, Spinner} from "@chakra-ui/react";
import BlogType from "../types/BlogType.ts";

const Blogs = () => {
    const [blogs, setBlogs] = useState<BlogType[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        document.title = "Blogs";
        setLoading(true);
        send_request("/blogs", "GET")
            .then(response => {
                setBlogs(response);
                setLoading(false);
            })
            .catch(error => {
                console.error('Error fetching blogs:', error);
            });
    }, []);

    return (
        <Box p={4}>
            <Heading as="h1" mb={4}>Blogs</Heading>
            {loading ? (
                <Spinner size="xl" />
            ) : (
                <SimpleGrid columns={{ sm: 1, md: 2, lg: 3 }} spacing={6}>
                    {blogs.map((blog) => (
                        <Blog key={blog.id}
                                 id={blog.id}
                                 description={blog.description}
                                 title={blog.title}
                        />
                    ))}
                </SimpleGrid>
            )}
        </Box>
    );
};

export default Blogs;
