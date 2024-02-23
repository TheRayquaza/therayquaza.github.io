import {useContext, useState} from 'react';
import {Box, Heading, Text, IconButton, Editable, EditableInput, EditablePreview, Link} from '@chakra-ui/react';
import { DeleteIcon, AddIcon, EditIcon, ViewIcon } from '@chakra-ui/icons';
import BlogType from '../types/BlogType';
import {GlobalContext} from "../context/GlobalProvider.tsx";
import {useNavigate} from "react-router";

type BlogProps = {
    blog : BlogType
    onDelete: (number:number) => void;
    onSave: (blog:BlogType) => void;
}

const Blog = ({ blog, onDelete, onSave } : BlogProps) => {
    const { isAdmin } = useContext(GlobalContext);
    const [editedTitle, setEditedTitle] = useState<string>(blog.title);
    const [editedDescription, setEditedDescription] = useState<string>(blog.description);

    let navigate = useNavigate();

    const handleSaveClick = () => {
        blog.title = editedTitle;
        blog.description = editedDescription;
        onSave(blog);
    }

    if (isAdmin)
        return (
            <Box p={4} shadow="md" borderWidth="1px" rounded="md" color="gray.700">
                <IconButton icon={<EditIcon/>} onClick={handleSaveClick} aria-label="Save" size="sm"/>
                <IconButton icon={<DeleteIcon/>} onClick={() => onDelete(blog.id)} aria-label="Delete" size="sm"/>
                <IconButton icon={<ViewIcon/>} size="sm" onClick={(e) => navigate(`/blogs/${blog.id}`)}/>
                <Editable defaultValue={editedTitle} onChange={setEditedTitle}>
                    <EditablePreview/>
                    <EditableInput/>
                </Editable>
                <Editable defaultValue={editedDescription} onChange={setEditedDescription}>
                    <EditablePreview/>
                    <EditableInput/>
                </Editable>
            </Box>
        );

    return (
        <Box p={4} shadow="md" borderWidth="1px" rounded="md" color="gray.700">
            <IconButton icon={<ViewIcon/>} size="sm" aria-label="View" onClick={(e) => navigate(`/blogs/${blog.id}`)}/>
            <Heading>{blog.title}</Heading>
            <Text>{blog.description}</Text>
        </Box>
    );
};

export default Blog;
