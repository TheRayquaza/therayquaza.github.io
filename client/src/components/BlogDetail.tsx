import {useState, useEffect, useContext} from 'react';
import { useParams, Link } from 'react-router-dom';
import { toast } from "react-toastify";
import { Button, Box, Spinner } from "@chakra-ui/react";
import BlogContentType from "../types/BlogContentType.ts";
import { send_request } from "../scripts/request.ts";
import BlogContentDetail from "./BlogContentDetail.tsx";
import BlogType from "../types/BlogType.ts";
import {ArrowBackIcon} from "@chakra-ui/icons";
import {GlobalContext} from "../context/GlobalProvider.tsx";

const BlogDetail = () => {
    const { isAdmin, apiKey } = useContext(GlobalContext);
    const { blogId } = useParams();
    const [blog, setBlog] = useState<BlogType>(null);
    const [contents, setContents] = useState<BlogContentType[]>([]);

    useEffect(() => {
        Promise.all([ fetchBlogData(), fetchContentsData() ]).then(() => document.title = `${blog.title}`)
    }, [blogId]);

    const fetchBlogData = async () => {
        const response = await send_request(`/blogs/${blogId}`, "GET");
        if (response && response.error)
            toast.error(response.error);
        else
            setBlog(response);
    };

    const fetchContentsData = async () => {
        const response = await send_request(`/blogs/${blogId}/content`, "GET");
        if (response && response.error)
            toast.error(response.error);
        else
            setContents(response.sort((a, b) => a.number - b.number));
    };

    const handleAddContent = async () => {
        const response = await send_request(
            `/blogs/${blogId}/content`,
            "POST",
            {
                "Content-Type": "application/json",
                "X-api-key": apiKey
            },
            {
                content: "New Text",
                type: "TEXT"
            }
        );
        if (response && response.error)
            toast.error(response.error);
        else
            await fetchContentsData();
    };

    const saveContent = async (content:BlogContentType) => {
        const response = await send_request(
            `/blogs/${content.blogId}/content/${content.id}`,
            "PUT",
            {
                "Content-Type" : "application/json",
                "X-api-key" : apiKey
            },
            {
                "content" : content.content,
                "type" : content.type,
                "number" : content.number
            }
        )
        if (response && response.error)
            toast.error(response.error)
    }

    const handleMoveContent = async (blogContent: BlogContentType, direction: 'up' | 'down') => {
        const index = contents.findIndex(b => b.id == blogContent.id)
        if ((index == 0 && direction === 'up') || (index == contents.length - 1 && direction === 'down'))
            return;
        const newIndex = direction === 'up' ? index - 1 : index + 1;
        const temp = contents[index].number;
        contents[index].number = contents[newIndex].number;
        contents[newIndex].number = temp;

        await Promise.all([
            saveContent(contents[index]),
            saveContent(contents[newIndex])
        ]);
        await fetchContentsData();
    };


    return (
        <Box textAlign="center" p="4">
            <Link to="/blogs/">
                <Button colorScheme="blue" size="sm" mb="2" leftIcon={<ArrowBackIcon />} mr="2">Back</Button>
            </Link>
            <h1 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '16px' }}>{blog?.title}</h1>
            {
                isAdmin ? (
                <Box display="flex" justifyContent="center" mb="2">
                    <Button onClick={handleAddContent} colorScheme="blue" size="sm" mr="2">Add Content</Button>
                    <Button onClick={fetchContentsData} colorScheme="teal" size="sm">Refresh Content</Button>
                </Box>
                ) : null
            }
            {
                contents.map(content => (
                <BlogContentDetail
                    key={content.id}
                    content={content}
                    onUpdate={fetchContentsData}
                    onMove={handleMoveContent}
                />
                ))
            }
            {blog ? null : <Spinner size="xl"/>}
        </Box>
    );
};

export default BlogDetail;
