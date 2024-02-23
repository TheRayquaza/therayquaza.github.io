import { useEffect, useState } from "react";
import { Box, Heading, SimpleGrid, Spinner } from "@chakra-ui/react";
import Project from "./Project";
import ProjectType from "../types/ProjectType.ts";
import {send_request} from "../scripts/request.ts";

const ProjectsPage = () => {
    const [projects, setProjects] = useState<ProjectType[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        document.title = "Admin - Projects";
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

    if (loading)
        return <Spinner size="xl" />

    return (
        <Box p={4}>
            <Heading as="h1" mb={4}>Projects</Heading>
                <SimpleGrid columns={{ sm: 1, md: 2, lg: 3 }} spacing={6}>
                    {projects.map((project) => <Project key={project.id} project={project}/>)}
                </SimpleGrid>
        </Box>
    );
};

export default ProjectsPage;
