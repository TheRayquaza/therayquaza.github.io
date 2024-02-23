package mlspot.backend.errors;

import lombok.Data;

@Data
public abstract class Error {
    String error;
    Integer status;

    public Error(String error, Integer status) {
        this.error = error;
        this.status = status;
    }
}
