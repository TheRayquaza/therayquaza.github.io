package mlspot.backend.presentation.rest.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.With;

import java.util.List;

@Data
@With
@NoArgsConstructor
@AllArgsConstructor
public class BlogCategoryStructureResponse {
    String name;
    Long id;
    List<BlogCategoryStructureResponse> children;
}
