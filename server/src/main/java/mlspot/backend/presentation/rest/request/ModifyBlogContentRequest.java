package mlspot.backend.presentation.rest.request;

import lombok.Data;

@Data
public class ModifyBlogContentRequest {
    String content;
    String type;
    Long number;
}
