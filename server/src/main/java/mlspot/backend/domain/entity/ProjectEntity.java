package mlspot.backend.domain.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.With;

import java.util.ArrayList;
import java.util.List;

@With
@Data
@AllArgsConstructor
@NoArgsConstructor
public class ProjectEntity {
    Long id;
    String name = null;
    String description = null;
    List<String> technologies = new ArrayList<>();
    String startingDate = null;
    String finishedData = null;
    Long members = 1L;
    String link = "";
}
