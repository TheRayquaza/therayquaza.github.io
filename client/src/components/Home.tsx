import {Box, Heading, Link, SimpleGrid} from "@chakra-ui/react";
import { Link as RouterLink } from "react-router-dom";
import backgroundImage from "/images/jelly_fish.jpg";
import { useEffect } from "react";
import Footer from "./Footer.tsx";

const Home = () => {
    useEffect(() => {
        document.title = "ML Spot";
    }, []);

    return (
        <Box
            width="100%"
            height="100vh"
            overflow="hidden"
            backgroundImage={`url(${backgroundImage})`}
            backgroundSize="cover"
            backgroundPosition="center"
            backgroundRepeat="no-repeat"
            display="flex"
            flexDirection="column"
            justifyContent="space-between"
            position="relative"
        >
            <Box
                height="90%"
                width="100%"
                display="flex"
                alignItems="center"
                justifyContent="center"
                flexDirection="column"
            >
                <SimpleGrid columns={4} spacing={10}>
                    <Box>
                        <a href="/pdf/CV_Mateo_Lelong_latest.pdf" color="white">
                            CV
                        </a>
                    </Box>
                    <Box>
                        <Link as={RouterLink} to="/blogs" mr={4} color="white">
                            Blogs
                        </Link>
                    </Box>
                    <Box>
                        <Link as={RouterLink} to="/tools" mr={4} color="white">
                            Tools
                        </Link>
                    </Box>
                    <Box>
                        <Link as={RouterLink} to="/projects" color="white">
                            Projects
                        </Link>
                    </Box>
                </SimpleGrid>
            </Box>
            <Box
                height="10%"
                width="100%"
                backgroundColor="deepblue"
                color="white"
                display="flex"
                justifyContent="center"
                alignItems="center"
            >
                <Footer />
            </Box>
        </Box>
    );
};

export default Home;
