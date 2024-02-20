package mlspot.backend.utils;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DateUtil {
    public static boolean isValidLocalDate(String date) {
        String regex = "^\\d{4}-\\d{2}-\\d{2}$";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(date);
        return matcher.matches();
    }
}
