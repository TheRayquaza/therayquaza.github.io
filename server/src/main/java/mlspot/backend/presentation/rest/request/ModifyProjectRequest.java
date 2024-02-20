package mlspot.backend.presentation.rest.request;

import lombok.Data;

import java.util.List;

@Data
public class ModifyProjectRequest {
    String name;
    String description;
    List<String> technologies;
    String startingDate;
    String finishedDate;
    Long members;
    String link;
}
