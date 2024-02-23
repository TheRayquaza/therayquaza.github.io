import {ReactNode, useContext, useState} from 'react';
import { Box, Button, Grid, IconButton, Image, Select, Text, Textarea } from "@chakra-ui/react";
import { CheckIcon, DeleteIcon, ChevronUpIcon, ChevronDownIcon } from "@chakra-ui/icons";
import { send_request } from "../scripts/request.ts";
import { toast } from "react-toastify";
import BlogContentType from "../types/BlogContentType.ts";
import {GlobalContext} from "../context/GlobalProvider.tsx";

interface BlogContentDetailProps {
    content: BlogContentType;
    onMove: (content: BlogContentType, direction: 'up' | 'down') => void;
}

const BlogContentDetail = ({ content, onUpdate, onMove }: BlogContentDetailProps) => {
    const { isAdmin, apiKey } = useContext(GlobalContext);
    const [editedContent, setEditedContent] = useState<BlogContentType>(content);
    const [isPreviewMode, setIsPreviewMode] = useState<boolean>(true);

    const handleContentChange = (value: string) => {
        setEditedContent(prevState => ({
            ...prevState,
            content: value
        }));
    };

    const handleSave = async () => {
        if (isPreviewMode) {
            setIsPreviewMode(false);
            return;
        }
        const response = await send_request(
            `/blogs/${content.blogId}/content/${content.id}`,
            "PUT",
            {
                "Content-Type": "application/json",
                "X-api-key": apiKey
            },
            {
                "type": editedContent.type,
                "content": editedContent.content,
                "number": editedContent.number
            }
        );
        if (response && response.error) {
            toast.error(response.error);
        } else {
            toast.success("Changes saved successfully.");
            setIsPreviewMode(true);
        }
    };

    const handleDeleteContent = async () => {
        const response = await send_request(
            `/blogs/${content.blogId}/content/${content.id}`,
            "DELETE",
            {
                "Content-Type": "application/json",
                "X-api-key": apiKey
            }
        );
        if (response && response.error)
            toast.error(response.error);
        else
            onUpdate()
    };

    const handleTogglePreviewMode = () => {
        setIsPreviewMode(!isPreviewMode);
    };

    const handleTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        setEditedContent(prevState => ({
            ...prevState,
            type: e.target.value
        }));
    };

    const renderContent = () => {
        switch (editedContent.type) {
            case "IMAGE":
                return <Image src={editedContent.content} alt="Preview" />;
            case "TEXT":
                return <Text>{editedContent.content}</Text>;
            default:
                return <Text>{editedContent.content}</Text>;
        }
    };

    return (
        <Box mb="1rem">
            {isAdmin ?
                (
                <Grid templateColumns="7% 53% 20% 20%" gap="1rem" alignItems="center">
                    <Box/>
                    <Box>
                        {isPreviewMode ? renderContent() : (
                            <Textarea defaultValue={editedContent.content} onChange={e => handleContentChange(e.target.value)} />
                        )}
                    </Box>
                    <Box textAlign="right">
                        <Text>{content.number}</Text>
                        <Select value={editedContent.type} onChange={handleTypeChange} size="sm" disabled={isPreviewMode} w="100%">
                            <option value="TEXT">Text</option>
                            <option value="LINK">Link</option>
                            <option value="IMAGE">Image</option>
                            <option value="VIDEO">Video</option>
                        </Select>
                        {!isPreviewMode && (
                            <>
                                <IconButton
                                    icon={<ChevronUpIcon />}
                                    colorScheme="blue"
                                    aria-label="Move Up"
                                    size="sm"
                                    onClick={() => onMove(editedContent, 'up')}
                                    mt="0.5rem"
                                />
                                <IconButton
                                    icon={<ChevronDownIcon />}
                                    colorScheme="blue"
                                    aria-label="Move Down"
                                    size="sm"
                                    onClick={() => onMove(editedContent, 'down')}
                                    mt="0.5rem"
                                />
                                <IconButton
                                    icon={<CheckIcon />}
                                    colorScheme="green"
                                    aria-label="Save"
                                    size="sm"
                                    onClick={handleSave}
                                    mt="0.5rem"
                                />
                                <IconButton
                                    icon={<DeleteIcon />}
                                    colorScheme="red"
                                    aria-label="Delete"
                                    size="sm"
                                    onClick={handleDeleteContent}
                                    mt="0.5rem"
                                />
                            </>
                        )}
                        <Button
                            colorScheme={isPreviewMode ? "blue" : "teal"}
                            size="sm"
                            onClick={handleTogglePreviewMode}
                            mt="0.5rem"
                        >
                            {isPreviewMode ? "Edit" : "Preview"}
                        </Button>
                    </Box>
                    <Box/>
                </Grid>
            ) : (
                <Grid templateColumns="10% 80% 10%" gap="1rem" alignItems="center">
                    <></>
                    <Box>
                        {renderContent()}
                    </Box>
                    <></>
                </Grid>
            )}
        </Box>
    );
};

export default BlogContentDetail;
