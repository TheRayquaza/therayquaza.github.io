import {useContext, useEffect, useState} from "react";
import { Box, Heading, SimpleGrid, Spinner, Button } from "@chakra-ui/react";
import Project from "./Project";
import ProjectType from "../types/ProjectType.ts";
import { send_request } from "../scripts/request.ts";
import {toast} from "react-toastify";
import {GlobalContext} from "../context/GlobalProvider.tsx";

const Projects = () => {
    const { apiKey, isAdmin } = useContext(GlobalContext);
    const [projects, setProjects] = useState<ProjectType[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchProjects = async () => {
        setLoading(true);
        const response = await send_request("/projects", "GET");
        if (response && response.error)
            toast.error(response.error)
        else
            setProjects(response.sort((a, b) => a.id - b.id));
        setLoading(false);
    };

    const createProject = async () => {
        setLoading(true);
        const response = await send_request(
            "/projects",
            "POST",
            {
                "Content-Type" : "application/json",
                "X-api-key" : apiKey
            },
            {
                "name" : "new name"
            }
        );
        if (response && response.error)
            toast.error(response.error)
        else
            await fetchProjects();
        setLoading(false);
    };

    const handleDelete = async (projectId: string) => {
        setProjects(prevProjects => prevProjects.filter(project => project.id !== projectId));
    };

    useEffect(() => {
        document.title = "Projects";
        fetchProjects();
    }, []);

    if (loading)
        return <Spinner size="xl" />;

    return (
        <Box p={4}>
            <Heading as="h1" mb={4}>Projects</Heading>
            {
                isAdmin ? (
                    <>
                        <Button mb={4} onClick={fetchProjects}>Refresh Projects</Button>
                        <Button mb={4} onClick={createProject}>Create New Project</Button>
                    </>
                ) : null
            }
            <SimpleGrid columns={{ sm: 1, md: 2, lg: 3 }} spacing={6}>
                {projects.map((project) => <Project key={project.id} project={project} onDelete={handleDelete}/>)}
            </SimpleGrid>
        </Box>
    );
};

export default Projects;
