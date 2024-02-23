package mlspot.backend.errors;

public class BadRequestError extends Error {

    public BadRequestError() {
        super("Bad request", 400);
    }
    public BadRequestError(String error, Integer status) {
        super(error, status);
    }
}
