import {useContext, useEffect, useState} from 'react';
import { send_request } from "../scripts/request.ts";
import Blog from "./Blog";
import { Box, Button, Heading, SimpleGrid, Spinner } from "@chakra-ui/react";
import BlogType from "../types/BlogType.ts";
import { toast } from "react-toastify";
import {GlobalContext} from "../context/GlobalProvider.tsx";

const Blogs = () => {
    const { isAdmin, apiKey } = useContext(GlobalContext);
    const [blogs, setBlogs] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (isAdmin)
            document.title = "Admin - Blogs";
        else
            document.title = "Blogs"
        setLoading(true);
        fetchBlogs();
    }, []);

    const fetchBlogs = async () => {
        const response = await send_request("/blogs", "GET");
        if (response && response.error)
            toast.error(response.error);
        else {
            setBlogs(response);
            setLoading(false);
        }
    };

    const reloadBlogs = async () => {
        setLoading(true);
        fetchBlogs();
    };

    const handleSave = async (blog: BlogType) => {
        try {
            await send_request(
                `/blogs/${blog.id}`,
                "PUT",
                {
                    "Content-Type": "application/json",
                    "X-api-key": apiKey
                },
                {
                    "title": blog.title,
                    "description": blog.description
                }
            );
            await reloadBlogs();
        } catch (error) {
            toast.error("Failed to save blog");
        }
    };

    const handleCreate = async () => {
        try {
            await send_request(
                "/blogs",
                "POST",
                {
                    "Content-Type": "application/json",
                    "X-api-key": apiKey
                },
                {
                    "title": "new blog"
                }
            );
            await reloadBlogs();
        } catch (error) {
            toast.error("Failed to create blog");
        }
    };

    const handleDelete = async (blogId) => {
        try {
            await send_request(
                `/blogs/${blogId}`,
                "DELETE",
                {
                    "Content-Type": "application/json",
                    "X-api-key": apiKey
                }
            );
            await reloadBlogs();
        } catch (error) {
            toast.error("Failed to delete blog");
        }
    };

    if (loading)
        return <Spinner size="xl"/>

    return (
        <Box p={4}>
            <Heading as="h1" mb={4}>Blogs</Heading>
            {
                isAdmin ? (
                    <Box>
                        <Button onClick={handleCreate} mb={4}>Create New Blog</Button>
                        <Button onClick={reloadBlogs} mb={4} ml={4}>Refresh</Button>
                    </Box>
                ) : null
            }
            <SimpleGrid columns={{ sm: 1, md: 2, lg: 3 }} spacing={6}>
                {blogs.map((blog) => (
                    <Blog key={blog.id}
                          blog={blog}
                          onDelete={handleDelete}
                          onSave={handleSave}
                    />
                ))}
            </SimpleGrid>
        </Box>
    );
};

export default Blogs;
