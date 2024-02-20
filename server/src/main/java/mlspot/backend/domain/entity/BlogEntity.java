package mlspot.backend.domain.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.With;

import java.util.ArrayList;
import java.util.List;

@Data
@With
@AllArgsConstructor
@NoArgsConstructor
public class BlogEntity {
    Long id;
    String title = "";
    String description = "";
}
