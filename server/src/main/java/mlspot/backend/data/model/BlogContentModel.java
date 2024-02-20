package mlspot.backend.data.model;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import lombok.*;

import javax.persistence.*;

@Getter
@Setter
@Entity
@With
@AllArgsConstructor
@NoArgsConstructor
@Table(name = "blog_content")
public class BlogContentModel extends PanacheEntityBase {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY) Long id;
    @Column(length = 8192) String content;
    String type;
    Long blogId;
}
