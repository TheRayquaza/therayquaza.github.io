package mlspot.backend.presentation.rest.request;

import lombok.Data;

@Data
public class CreateBlogContentRequest {
    String type;
    String content;
}
