package mlspot.backend.domain.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.With;

@Data
@With
@AllArgsConstructor
@NoArgsConstructor
public class BlogContentEntity {
    Long id;
    BlogContentEnumEntity blogContentEnumEntity;
    String content;
    Long number;
    Long blogId;
}
