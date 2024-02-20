package mlspot.backend.presentation.rest.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.With;

@Data
@With
@NoArgsConstructor
@AllArgsConstructor
public class BlogResponse {
    Long id;
    String title;
    String description;
}
