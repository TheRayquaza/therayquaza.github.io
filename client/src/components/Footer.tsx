import { Box, Icon, Link, Text } from "@chakra-ui/react";
import { FaGithub, FaLinkedin } from "react-icons/fa";
import {MdEmail} from "react-icons/md";

const Footer = () => {
    return (
        <Box
            as="footer"
            backgroundColor="deepblue"
            color="white"
            textAlign="center"
            py={6}
        >
            <Text fontSize="lg" mb={4}>
                Connect with me
            </Text>
            <Box display="flex" justifyContent="center" alignItems="center">
                <Link href="https://github.com/TheRayquaza" isExternal mx={4}>
                    <FaGithub />
                </Link>
                <Link href="https://linkedin.com/in/matéo-lelong-4b05ba256" isExternal mx={4}>
                    <FaLinkedin/>
                </Link>
                <Link href="mailto:mateo.lelong@gmail.com" mx={4}>
                    <MdEmail />
                </Link>
            </Box>
            <Text mt={4} fontSize="sm">
                © {new Date().getFullYear()} Mateo LELONG. All rights reserved.
            </Text>
        </Box>
    );
};

export default Footer;
