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
public class ProjectResponse {
    Long id;
    String name;
    String description;
    List<String> technologies;
    String startingDate;
    String finishedDate;
    Long members;
    String link;
}
