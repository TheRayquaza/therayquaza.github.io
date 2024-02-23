package mlspot.backend.errors;

public class UnauthorizedError extends Error {

    public UnauthorizedError(String error, Integer status) {
        super(error, status);
    }

    public UnauthorizedError() {
        super("Unauthorized access to this resource", 401);
    }
}
