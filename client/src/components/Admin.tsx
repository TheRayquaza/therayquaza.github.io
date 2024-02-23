import { useState, useContext } from 'react';
import { GlobalContext } from '../context/GlobalProvider.tsx';
import { Box, Button, Input, Center, Card, CardBody } from '@chakra-ui/react';
import { send_request } from "../scripts/request.ts";
import { toast } from "react-toastify";
import {useNavigate} from "react-router";

const Admin = () => {
    const { setApiKey, setIsAdmin } = useContext(GlobalContext);
    const [apiKeyInput, setApiKeyInput] = useState('');

    let navigate = useNavigate();

    const handleApiKeyChange = (event) => {
        setApiKeyInput(event.target.value);
    };

    const handleSave = async () => {
        const response = await send_request(
            "/admin",
            "POST",
            {
                "Content-Type": "application/json"
            },
            {
                "apiKey": apiKeyInput
            }
        );
        if (response && response.error)
            toast.error(response.error);
        else {
            toast.success("Now logged in as admin");
            setApiKey(apiKeyInput);
            setIsAdmin(true);
            navigate("/");
        }
    };

    const handleKeyPress = (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            handleSave();
        }
    };

    return (
        <Center h="100vh">
            <Card p={4} maxW="400px">
                <CardBody>
                    <Input
                        type="text"
                        value={apiKeyInput}
                        onChange={handleApiKeyChange}
                        placeholder="Enter the API key"
                        mb={4}
                    />
                    <Center>
                        <Button colorScheme="teal" onClick={handleSave}>
                            Save
                        </Button>
                    </Center>
                </CardBody>
            </Card>
        </Center>
    );
};

export default Admin;
