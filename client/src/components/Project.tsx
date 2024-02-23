import { Box, Heading, Text, Link as ChakraLink } from "@chakra-ui/react";
import ProjectType from "../types/ProjectType";
import {GlobalContext} from "../context/GlobalProvider.tsx";
import {useContext} from "react";

type ProjectProps = {
    project:ProjectType;
}

const Project = ( { project } : ProjectProps ) => {
    const { isAdmin, apiKey } = useContext(GlobalContext);

    return (
        <Box p={4} shadow="md" borderWidth="1px" rounded="md" bg="dark">
            <Heading as="h3" size="md" color="blue.600" mb={2}>
                {project.name}
            </Heading>
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
        </Box>
    );
};

export default Project;
