package mlspot.backend.errors;

import lombok.Data;

public class NotFoundError extends Error {
    public NotFoundError(String error, Integer status) {
        super(error, status);
    }

    public NotFoundError() {
        super("This resource could not be found", 404);
    }
}
