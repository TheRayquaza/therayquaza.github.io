import React, {useContext, useState} from "react";
import { Box, Heading, Text, Link as ChakraLink, Input, Textarea, Button } from "@chakra-ui/react";
import ProjectType from "../types/ProjectType";
import { GlobalContext } from "../context/GlobalProvider.tsx";
import { send_request } from "../scripts/request.ts";
import { toast } from "react-toastify";

type ProjectProps = {
    project: ProjectType;
    onDelete: (projectId: string) => void;
}

const Project = ({ project, onDelete }: ProjectProps) => {
    const { isAdmin, apiKey } = useContext(GlobalContext);
    const [editableProject, setEditableProject] = useState<ProjectType>(project);

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = e.target;
        setEditableProject(prevState => ({
            ...prevState,
            [name]: value
        }));
    };

    const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        setEditableProject(prevState => ({
            ...prevState,
            [name]: value
        }));
    };

    const handleSaveChanges = async () => {
        try {
            const response = await send_request(
                `/projects/${project.id}`,
                "PUT",
                {
                    "Content-Type": "application/json",
                    "X-api-key": apiKey
                },
                {
                    "name": editableProject.name,
                    "description": editableProject.description,
                    "technologies": editableProject.technologies,
                    "startingDate": editableProject.startingDate,
                    "finishedDate": editableProject.finishedDate,
                    "members": editableProject.members,
                    "link": editableProject.link
                }
            );
            if (response && response.error)
                toast.error(response.error);
            else
                setEditableProject(response);
        } catch (error) {
            console.error("Error saving changes:", error);
            toast.error("An error occurred while saving changes");
        }
    };

    const handleDelete = async () => {
        const response = await send_request(
            `/projects/${project.id}`,
            "DELETE",
            {
                "X-api-key": apiKey

            }
        );
        if (response && response.error)
            toast.error(response.error);
        else
            onDelete(project.id as string);
    };

    return (
        <Box p={4} shadow="md" borderWidth="1px" rounded="md" bg="dark">
            <Heading as="h3" size="md" color="blue.600" mb={2}>
                {isAdmin ? (
                    <Input
                        type="text"
                        name="name"
                        value={editableProject.name}
                        onChange={handleInputChange}
                    />
                ) : (
                    project.name
                )}
            </Heading>
            {isAdmin ? (
                <>
                    <Text fontSize="sm" mb={2}>Starting Date:</Text>
                    <Input
                        type="text"
                        name="startingDate"
                        value={editableProject.startingDate}
                        onChange={handleInputChange}
                    />
                    <Text fontSize="sm" mb={2}>Finished Date:</Text>
                    <Input
                        type="text"
                        name="finishedDate"
                        value={editableProject.finishedDate}
                        onChange={handleInputChange}
                    />
                    <Text fontSize="sm" mb={2}>Description:</Text>
                    <Textarea
                        name="description"
                        value={editableProject.description}
                        onChange={handleTextareaChange}
                    />
                    <Text fontSize="sm" mb={2}>Link:</Text>
                    <Input
                        type="text"
                        name="link"
                        value={editableProject.link}
                        onChange={handleInputChange}
                    />
                    <Button onClick={handleSaveChanges}>Save</Button>
                    <Button onClick={handleDelete} ml={2} colorScheme="red" size="sm">Delete</Button>
                </>
            ) : (
                <>
                    <Text fontSize="sm" mb={2}>
                        {new Date(project.startingDate).toLocaleDateString()} - {new Date(project.finishedDate).toLocaleDateString()}
                    </Text>
                    <Text mb={4}>{project.description}</Text>
                    <Text mb={2} fontSize="sm" color="blue.700">
                        Technologies:{" "}
                        {project.technologies.map((tech, index) => (
                            <span key={index}>{tech}{index < project.technologies.length - 1 ? ", " : ""}</span>
                        ))}
                    </Text>
                    {project.link && (
                        <ChakraLink href={project.link} color="blue.500" isExternal>
                            View Project
                        </ChakraLink>
                    )}
                </>
            )}
        </Box>
    );
};

export default Project;
