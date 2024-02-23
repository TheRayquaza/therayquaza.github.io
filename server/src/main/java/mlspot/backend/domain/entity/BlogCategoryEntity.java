package mlspot.backend.domain.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.With;

@Data
@With
@AllArgsConstructor
@NoArgsConstructor
public class BlogCategoryEntity {
    Long id;
    Long parentId = -1L;
    String name;
}
