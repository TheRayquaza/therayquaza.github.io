import { useEffect } from 'react';
import { useNavigate } from "react-router";
import { Box, Text, Spinner } from '@chakra-ui/react';

const Default = () => {
    const navigate = useNavigate();

    useEffect(() => {
        document.title = "Page not found";

        const timeout = setTimeout(() => {
            navigate('/');
        }, 3000);

        return () => clearTimeout(timeout);
    }, [navigate]);

    return (
        <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            justifyContent="center"
            height="100vh"
        >
            <Text fontSize="6xl" color="teal.500" marginBottom={4}>
                404
            </Text>
            <Text fontSize="2xl" textAlign="center" marginBottom={4}>
                Sorry, we couldn't find the page you were looking for.
            </Text>
            <Text fontSize="lg" textAlign="center" marginBottom={4}>
                You will be redirected to the home page in 3 seconds.
            </Text>
            <Spinner size="xl" color="teal.500" />
        </Box>
    );
};

export default Default;