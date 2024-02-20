import React from 'react'
import ReactDOM from 'react-dom/client'
import {ChakraProvider, ColorModeScript, CSSReset} from "@chakra-ui/react"
import theme from './theme.tsx'
import App from './App.tsx'


ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
    <React.StrictMode>
        <ChakraProvider theme={theme}>
            <CSSReset />
            <ColorModeScript initialColorMode={theme.config.initialColorMode}/>
            <App/>
        </ChakraProvider>
    </React.StrictMode>
)
