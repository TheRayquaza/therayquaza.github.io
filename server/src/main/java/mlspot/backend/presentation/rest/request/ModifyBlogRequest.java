package mlspot.backend.presentation.rest.request;

import lombok.Data;

@Data
public class ModifyBlogRequest {
    String title;
    String description;
}
