package mlspot.backend.presentation.rest.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.With;

@With
@AllArgsConstructor
@Data
@NoArgsConstructor
public class BlogContentResponse {
    Long id;
    BlogContentEnumResponse type;
    String content;
}
