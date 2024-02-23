package mlspot.backend.domain.entity;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.With;

@Data
@With
@AllArgsConstructor
@NoArgsConstructor
public class BlogCategoryStructureEntity {
    List<BlogCategoryStructureEntity> children;
    String name;
    Long id;
}
