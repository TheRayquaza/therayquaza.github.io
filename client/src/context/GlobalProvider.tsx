import {useState, createContext, Dispatch, SetStateAction, ReactNode} from "react";

type GlobalContextType = {
    apiKey : string;
    isAdmin : boolean;
    setIsAdmin:  Dispatch<SetStateAction<boolean>>;
    setApiKey:  Dispatch<SetStateAction<string>>;
};

const initialState: GlobalContextType = {
    apiKey : "",
    isAdmin : false,
    setIsAdmin : () => {},
    setApiKey : () => {}
};

export const GlobalContext = createContext<GlobalContextType>(initialState);

const GlobalProvider = ({ children }: { children: ReactNode }) => {
    const [apiKey, setApiKey] = useState<string>("");
    const [isAdmin, setIsAdmin] = useState<boolean>(false);

    return (
        <GlobalContext.Provider value={{ apiKey, isAdmin, setIsAdmin, setApiKey }}>
            {children}
        </GlobalContext.Provider>
    );
};

export default GlobalProvider;