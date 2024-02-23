package mlspot.backend.data.model;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import lombok.*;

import javax.persistence.*;
import java.time.LocalDate;

@Getter
@Setter
@With
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "project")
public class ProjectModel extends PanacheEntityBase {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    Long id;
    String name;
    String description = "";
    String technologies = "";
    LocalDate startingDate = null;
    LocalDate finishedDate = null;
    Long members = 1L;
    String link = "";
}
