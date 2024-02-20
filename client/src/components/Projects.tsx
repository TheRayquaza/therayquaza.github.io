import React, { useEffect, useState } from "react";
import { Box, Heading, SimpleGrid, Spinner } from "@chakra-ui/react";
import Project from "./Project";
import ProjectType from "../types/ProjectType.ts";
import {send_request} from "../scripts/request.ts";

const ProjectsPage = () => {
    const [projects, setProjects] = useState<ProjectType[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        document.title = "Projects";
        send_request("/projects", "GET")
            .then((data) => {
                console.log(data)
                setProjects(data);
                setLoading(false);
            })
            .catch((error) => {
                console.error("Error fetching projects:", error);
                setLoading(false);
            });
    }, []);

    return (
        <Box p={4}>
            <Heading as="h1" mb={4}>Projects</Heading>
            {loading ? (
                <Spinner size="xl" />
            ) : (
                <SimpleGrid columns={{ sm: 1, md: 2, lg: 3 }} spacing={6}>
                    {projects.map((project) => (
                        <Project key={project.id}
                                 id={project.id}
                                 description={project.description}
                                 technologies={project.technologies}
                                 finishedDate={project.finishedDate}
                                 startingDate={project.startingDate}
                                 name={project.name}
                        />
                    ))}
                </SimpleGrid>
            )}
        </Box>
    );
};

export default ProjectsPage;
