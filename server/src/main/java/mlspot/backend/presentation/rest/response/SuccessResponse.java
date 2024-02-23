package mlspot.backend.presentation.rest.response;

import lombok.Data;

@Data
public class SuccessResponse {
    String success = "operation was successful";
    Integer status = 200;
}
