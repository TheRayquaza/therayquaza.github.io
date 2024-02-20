const url : string = "http://localhost:8082";

export const send_request = async (endpoint : string, method: string, headers : RequestInit["headers"] = {}, body : any = null) : Promise<any> => {
    if (!body) {
        return await fetch(url + endpoint, {
            method: method,
            headers : headers
        }).then(async (res)  => await res.json());
    }
    return await fetch(url + endpoint, {
        method: method,
        headers : headers,
        body: JSON.stringify(body)
    }).then(async (res)  => await res.json());
}