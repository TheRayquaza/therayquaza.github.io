package mlspot.backend.presentation.rest.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.With;

@Data
@With
@AllArgsConstructor
@NoArgsConstructor
public class BlogCategoryResponse {
    Long id;
    Long parentId;
    String name;
}
