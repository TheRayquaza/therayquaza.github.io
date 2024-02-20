import { Box } from "@chakra-ui/react";

const BackgroundImage = ({ imageUrl, children }) => {
    return (
        <Box
            bg={`url(${imageUrl})`}
            backgroundSize="cover"
            backgroundPosition="center"
            backgroundRepeat="no-repeat"
            width="100vw"
            height="100vh"
            display="flex"
            justifyContent="center"
            alignItems="center"
        >
            {children}
        </Box>
    );
};

export default BackgroundImage;
