import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';

import 'react-toastify/dist/ReactToastify.css';

import Home from './components/Home';
import Default from './components/Default';
import Projects from './components/Projects.tsx'
import Blogs from './components/Blogs.tsx'
import theme from "./theme.tsx";
import {ChakraProvider} from "@chakra-ui/react";

const App = () => {
    return (
        <BrowserRouter>
            <ChakraProvider theme={theme}>
            <ToastContainer limit={5} pauseOnHover={false} autoClose={1500}/>
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="*" element={<Default />} />
                <Route path="/projects" element={<Projects/>}/>
                <Route path="/blogs" element={<Blogs/>}/>
            </Routes>
            </ChakraProvider>
        </BrowserRouter>
    );
}

export default App;
