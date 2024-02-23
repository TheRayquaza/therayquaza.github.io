package mlspot.backend.presentation.rest.request;

import lombok.Data;

@Data
public class ModifyBlogCategoryRequest {
    String name;
    Long parentId;
}
