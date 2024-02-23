import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';

import 'react-toastify/dist/ReactToastify.css';

import Home from './components/Home';
import Default from './components/Default';
import Projects from './components/Projects.tsx'
import Blogs from './components/Blogs.tsx'
import BlogDetail from "./components/BlogDetail.tsx";
import GlobalProvider from "./context/GlobalProvider.tsx";
import Admin from "./components/Admin.tsx";

const App = () => {
    return (
        <GlobalProvider>
            <BrowserRouter>
                <ToastContainer limit={5} pauseOnHover={false} autoClose={1500}/>
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="*" element={<Default />} />
                    <Route path="/projects" element={<Projects/>}/>
                    <Route path="/blogs" element={<Blogs/>}/>
                    <Route path="/blogs/:blogId" element={<BlogDetail />} />
                    <Route path="/admin" element={<Admin/>}/>
                </Routes>
            </BrowserRouter>
        </GlobalProvider>
    );
}

export default App;